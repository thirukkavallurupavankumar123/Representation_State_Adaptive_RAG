from __future__ import annotations

from collections import defaultdict, deque
import json
from pathlib import Path

import pytest

from sar_rag.final_evaluation import ANSWER_SCHEMA, logical_key, validate_answer
from sar_rag.generation_v1_1 import (
    MAX_COMPLETION_TOKENS, build_chat_payload_v1_1, build_messages_v1_1,
    request_fingerprint_v1_1,
)
from sar_rag.smoke_recovery import AttemptOutcome, load_jsonl
from sar_rag.smoke_v1_1 import V11SmokeRunner
from tools.run_smoke_generation_v1_1 import verify_authorization


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/final_evaluation"
V11 = OUTPUT / "generation_v1_1"


def valid_outcome(text: str = "Supported fact.") -> AttemptOutcome:
    payload = {
        "answerability": "ANSWERABLE",
        "claims": [{"text": text, "evidence_ids": ["E1"]}],
        "missing_information": "",
    }
    return AttemptOutcome(
        status="success", payload=payload, raw_response=json.dumps(payload),
        input_tokens=100, output_tokens=20, latency_ms=1,
    )


class ScriptedAnswerer:
    def __init__(self, scripts: dict[str, list[AttemptOutcome]]) -> None:
        self.scripts = {key: deque(values) for key, values in scripts.items()}
        self.order: list[str] = []

    def attempt(self, request: dict) -> AttemptOutcome:
        key = logical_key(request)
        self.order.append(key)
        return self.scripts[key].popleft()


def frozen_inputs() -> tuple[list[dict], dict, dict]:
    requests = load_jsonl(OUTPUT / "SMOKE_TEST_REQUESTS.jsonl")
    protocol = json.loads((V11 / "PROTOCOL.json").read_text(encoding="utf-8"))
    manifest = json.loads((V11 / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    return requests, protocol, manifest


def test_exact_outgoing_payload_and_strict_schema() -> None:
    requests, _, _ = frozen_inputs()
    payload = build_chat_payload_v1_1(requests[0]["question"], requests[0]["evidence"])
    assert set(payload) == {
        "model", "messages", "temperature", "top_p", "max_completion_tokens",
        "reasoning_effort", "response_format",
    }
    assert payload["model"] == "openai/gpt-oss-120b"
    assert payload["temperature"] == 0
    assert payload["top_p"] == 1
    assert payload["reasoning_effort"] == "medium"
    assert payload["max_completion_tokens"] == 1600
    assert "reasoning_format" not in payload
    response_format = payload["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"] == ANSWER_SCHEMA
    schema = response_format["json_schema"]["schema"]
    assert set(schema["properties"]) == set(schema["required"])
    assert schema["additionalProperties"] is False
    claim_schema = schema["properties"]["claims"]["items"]
    assert set(claim_schema["properties"]) == set(claim_schema["required"])
    assert claim_schema["additionalProperties"] is False


def test_prompt_explicitly_requires_missing_information_without_inserting_it() -> None:
    messages, _ = build_messages_v1_1("Question?", [{"evidence_id": "E1", "text": "Evidence"}])
    user = messages[-1]["content"]
    assert "exactly the three required top-level fields" in user
    assert "Always emit missing_information" in user
    assert "use an empty string when nothing is missing" in user
    with pytest.raises(ValueError, match="frozen schema"):
        validate_answer({"answerability": "ANSWERABLE", "claims": []}, 1)


def test_all_saved_valid_answers_still_pass_unchanged_semantic_validator() -> None:
    requests, _, _ = frozen_inputs()
    evidence_counts = {logical_key(row): len(row["evidence"]) for row in requests}
    saved = load_jsonl(OUTPUT / "paid_smoke_checkpoint/ANSWERS.jsonl")
    saved += load_jsonl(OUTPUT / "recovery_v1/checkpoint/ANSWERS.jsonl")
    assert len(saved) == 46
    for row in saved:
        payload = dict(row["response"])
        rendered = payload.pop("rendered_answer")
        assert validate_answer(payload, evidence_counts[row["logical_key"]])["rendered_answer"] == rendered


def test_saved_unresolved_generation_missing_required_field_is_rejected() -> None:
    payload = {
        "answerability": "PARTIALLY_ANSWERABLE",
        "claims": [{
            "text": "The average construction salary in the United Kingdom is £26,719.",
            "evidence_ids": ["E1"],
        }],
    }
    with pytest.raises(ValueError, match="frozen schema"):
        validate_answer(payload, 15)


def test_round_robin_records_first_and_eventual_validity(tmp_path: Path) -> None:
    requests, protocol, manifest = frozen_inputs()
    scripts = {logical_key(row): [valid_outcome(logical_key(row))] for row in requests}
    first_key, second_key = logical_key(requests[0]), logical_key(requests[1])
    scripts[first_key] = [
        AttemptOutcome(status="api_error", error="simulated json_validate_failed"),
        valid_outcome("Recovered first pair."),
    ]
    scripts[second_key] = [
        AttemptOutcome(
            status="success",
            payload={"answerability": "INSUFFICIENT", "claims": [{"text": "bad", "evidence_ids": ["E1"]}], "missing_information": "gap"},
            raw_response='{"simulated":"semantic failure"}',
        ),
        valid_outcome("Recovered second pair."),
    ]
    answerer = ScriptedAnswerer(scripts)
    summary = V11SmokeRunner(tmp_path, protocol, manifest).run(requests, answerer)
    assert answerer.order[:48] == [logical_key(row) for row in requests]
    assert answerer.order[48:] == [first_key, second_key]
    assert summary["cumulative_attempts"] == 50
    assert summary["first_attempt_valid_pairs"] == 46
    assert summary["eventually_valid_pairs"] == 48
    answers = load_jsonl(tmp_path / "ANSWERS.jsonl")
    by_key = {row["logical_key"]: row for row in answers}
    assert by_key[first_key]["first_attempt_valid"] is False
    assert by_key[first_key]["eventually_valid"] is True


def test_persistent_failure_remains_missing_after_three_attempts(tmp_path: Path) -> None:
    requests, protocol, manifest = frozen_inputs()
    failed_key = logical_key(requests[0])
    scripts = {logical_key(row): [valid_outcome()] for row in requests}
    scripts[failed_key] = [
        AttemptOutcome(status="api_error", error=f"simulated failure {number}")
        for number in range(1, 4)
    ]
    summary = V11SmokeRunner(tmp_path, protocol, manifest).run(
        requests, ScriptedAnswerer(scripts)
    )
    assert summary["cumulative_attempts"] == 50
    assert summary["eventually_valid_pairs"] == 47
    assert summary["persistent_failure_pairs"] == [failed_key]
    assert len([row for row in load_jsonl(tmp_path / "API_ATTEMPTS.jsonl") if row["logical_key"] == failed_key]) == 3


def test_authorization_template_cannot_execute() -> None:
    _, protocol, manifest = frozen_inputs()
    with pytest.raises(SystemExit, match="authorization content/fingerprint mismatch"):
        verify_authorization(V11 / "AUTHORIZATION_TEMPLATE.json", protocol, manifest)


def test_manifest_covers_all_pairs_and_fingerprints() -> None:
    requests, protocol, manifest = frozen_inputs()
    assert manifest["pair_count"] == 48
    assert manifest["maximum_total_api_attempts_including_retries"] == 144
    assert protocol["generation"]["decoding"]["max_output_tokens"] == MAX_COMPLETION_TOKENS
    expected = {logical_key(row): request_fingerprint_v1_1(row) for row in requests}
    assert {row["logical_key"]: row["request_fingerprint_v1_1"] for row in manifest["pairs"]} == expected


def test_protocol_fingerprint_drift_blocks_before_call(tmp_path: Path) -> None:
    requests, protocol, manifest = frozen_inputs()
    runner = V11SmokeRunner(tmp_path, protocol, manifest)
    protocol["generation"]["decoding"]["max_output_tokens"] += 1
    answerer = ScriptedAnswerer({logical_key(row): [valid_outcome()] for row in requests})
    with pytest.raises(RuntimeError, match="protocol fingerprint mismatch before call"):
        runner.run(requests, answerer)
    assert answerer.order == []
