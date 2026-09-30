from __future__ import annotations

from collections import defaultdict, deque
import json
from pathlib import Path

import pytest

from sar_rag.final_evaluation import logical_key
from sar_rag.smoke_recovery import AttemptOutcome, RecoveryRunner, load_jsonl, missing_requests
from tools.run_paid_smoke_recovery import retry_after_seconds, sanitize_error


GENERATION = {
    "model": "openai/gpt-oss-120b",
    "decoding": {
        "temperature": 0,
        "top_p": 1,
        "reasoning_effort": "medium",
        "max_output_tokens": 800,
        "response_format": "strict JSON schema",
    },
}
PRICING = {"input_per_million": 0.15, "output_per_million": 0.6}


def request(question_id: str, system: str = "traditional_k10") -> dict:
    return {
        "dataset": "fixture",
        "question_id": question_id,
        "system": system,
        "question": f"Question {question_id}?",
        "evidence": [{"evidence_id": "E1", "text": "Fixture evidence."}],
    }


def valid_outcome(text: str = "Supported fact.") -> AttemptOutcome:
    payload = {
        "answerability": "ANSWERABLE",
        "claims": [{"text": text, "evidence_ids": ["E1"]}],
        "missing_information": "",
    }
    return AttemptOutcome(
        status="success",
        payload=payload,
        raw_response=json.dumps(payload),
        input_tokens=100,
        output_tokens=20,
        latency_ms=5,
    )


class ScriptedAnswerer:
    def __init__(self, scripts: dict[str, list[AttemptOutcome]]) -> None:
        self.scripts = {key: deque(values) for key, values in scripts.items()}
        self.order: list[str] = []

    def attempt(self, item: dict, generation: dict) -> AttemptOutcome:
        assert generation == GENERATION
        key = logical_key(item)
        self.order.append(key)
        return self.scripts[key].popleft()


def make_runner(tmp_path: Path, items: list[dict], total: int = 6, per_pair: int = 3,
                maximum_cost_usd: float = 1.0) -> RecoveryRunner:
    return RecoveryRunner(
        tmp_path,
        "fixture-fingerprint",
        GENERATION,
        PRICING,
        {logical_key(item) for item in items},
        total,
        per_pair,
        maximum_cost_usd,
        {logical_key(item): 100 for item in items},
    )


def test_round_robin_retries_api_and_local_validation_failures(tmp_path: Path) -> None:
    items = [request("a"), request("b", "sar_rag_v1")]
    keys = [logical_key(item) for item in items]
    bad_local = AttemptOutcome(
        status="success",
        payload={
            "answerability": "INSUFFICIENT",
            "claims": [{"text": "Unsupported claim", "evidence_ids": ["E1"]}],
            "missing_information": "Not enough evidence.",
        },
        raw_response='{"bad":"semantic combination"}',
        input_tokens=100,
        output_tokens=20,
    )
    answerer = ScriptedAnswerer({
        keys[0]: [
            AttemptOutcome(status="api_error", error_type="RuntimeError", error="simulated 400"),
            valid_outcome("Recovered A."),
        ],
        keys[1]: [bad_local, valid_outcome("Recovered B.")],
    })
    runner = make_runner(tmp_path, items)
    summary = runner.run(items, answerer)

    assert answerer.order == [keys[0], keys[1], keys[0], keys[1]]
    assert summary["cumulative_attempts"] == 4
    assert summary["cumulative_answers"] == 2
    assert summary["missing_answers"] == 0
    failures = load_jsonl(tmp_path / "FAILURES.jsonl")
    assert [row["status"] for row in failures] == ["api_error", "local_validation_error"]
    assert failures[1]["raw_response"] == '{"bad":"semantic combination"}'


def test_global_attempt_ceiling_stops_before_extra_call(tmp_path: Path) -> None:
    items = [request("a"), request("b"), request("c")]
    scripts = {
        logical_key(item): [AttemptOutcome(status="api_error", error="simulated")]
        for item in items
    }
    answerer = ScriptedAnswerer(scripts)
    runner = make_runner(tmp_path, items, total=2, per_pair=3)
    summary = runner.run(items, answerer)
    assert len(answerer.order) == 2
    assert summary["cumulative_attempts"] == 2
    assert summary["attempt_ceiling_exhausted"] is True


def test_cost_ceiling_is_checked_before_every_call(tmp_path: Path) -> None:
    items = [request("a"), request("b")]
    keys = [logical_key(item) for item in items]
    one_attempt_ceiling = (100 * 0.15 + 800 * 0.6) / 1_000_000
    answerer = ScriptedAnswerer({key: [valid_outcome(key)] for key in keys})
    runner = make_runner(
        tmp_path, items, total=6, per_pair=3,
        maximum_cost_usd=one_attempt_ceiling,
    )
    summary = runner.run(items, answerer)
    assert len(answerer.order) == 1
    assert summary["cumulative_attempts"] == 1
    assert summary["cost_ceiling_exhausted"] is True
    attempt = load_jsonl(tmp_path / "API_ATTEMPTS.jsonl")[0]
    assert attempt["remaining_attempt_allowance_before_call"] == 6
    assert attempt["remaining_cost_allowance_before_call_usd"] == pytest.approx(one_attempt_ceiling)


def test_resume_does_not_duplicate_valid_answers(tmp_path: Path) -> None:
    items = [request("a"), request("b")]
    keys = [logical_key(item) for item in items]
    first = ScriptedAnswerer({key: [valid_outcome(key)] for key in keys})
    runner = make_runner(tmp_path, items)
    assert runner.run(items, first)["new_answers"] == 2
    resumed = make_runner(tmp_path, items)
    never_called = ScriptedAnswerer({})
    summary = resumed.run(items, never_called)
    assert summary["new_attempts"] == 0
    assert summary["cumulative_answers"] == 2


def test_rejects_request_set_drift(tmp_path: Path) -> None:
    items = [request("a"), request("b")]
    runner = make_runner(tmp_path, items)
    with pytest.raises(ValueError, match="authorized missing-pair set"):
        runner.run(items[:1], ScriptedAnswerer({}))


def test_real_checkpoint_has_exactly_eight_missing_pairs() -> None:
    root = Path(__file__).resolve().parents[1] / "output/final_evaluation"
    requests = load_jsonl(root / "SMOKE_TEST_REQUESTS.jsonl")
    answers = load_jsonl(root / "paid_smoke_checkpoint/ANSWERS.jsonl")
    missing = missing_requests(requests, answers)
    manifest = json.loads((root / "recovery_v1/RECOVERY_MANIFEST.json").read_text(encoding="utf-8"))
    assert len(missing) == 8
    assert {logical_key(row) for row in missing} == {
        row["logical_key"] for row in manifest["pairs"]
    }
    assert manifest["maximum_total_api_attempts_including_retries"] == 24
    assert manifest["configuration_change"] is False


def test_rate_limit_delay_parsing_and_error_sanitization() -> None:
    error = "Rate limit in organization `org_secret`; try again in 9015ms; gsk_not_a_real_key"
    assert retry_after_seconds(error) == pytest.approx(9.015)
    sanitized = sanitize_error(error)
    assert "org_secret" not in sanitized
    assert "gsk_not_a_real_key" not in sanitized
