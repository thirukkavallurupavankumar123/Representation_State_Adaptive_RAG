from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from sar_rag.final_evaluation import (
    AnswerResponse, CheckpointRunner, DeterministicFakeAnswerer, file_sha256,
    request_fingerprint, token_f1, validate_answer,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "final_evaluation"


def read_json(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def read_jsonl(name: str) -> list[dict]:
    return [json.loads(line) for line in (OUT / name).read_text(encoding="utf-8").splitlines() if line]


def generation() -> dict:
    protocol = read_json("FINAL_EVALUATION_PROTOCOL.json")
    return {
        "model": protocol["generation"]["model"],
        "decoding": {
            key: protocol["generation"][key]
            for key in ("temperature", "top_p", "reasoning_effort", "max_output_tokens", "response_format")
        },
    }


def tiny_request(system: str = "traditional_k10") -> dict:
    return {
        "dataset": "synthetic", "question_id": "synthetic:1", "system": system,
        "question": "What is supported?",
        "evidence": [{"evidence_id": "E1", "text": "The evidence supports alpha."}],
    }


def test_protocol_is_frozen_and_does_not_contain_a_secret() -> None:
    protocol = read_json("FINAL_EVALUATION_PROTOCOL.json")
    assert protocol["status"] == "FROZEN_BEFORE_ANY_PAID_EXECUTION"
    assert protocol["generation"]["model"] == "openai/gpt-oss-120b"
    assert protocol["generation"]["temperature"] == 0
    assert protocol["generation"]["max_output_tokens"] == 800
    assert protocol["smoke_test"]["maximum_calls"] == 48
    assert protocol["final_heldout"]["maximum_answer_calls"] == 18_855
    serialized = json.dumps(protocol).lower()
    assert "groq_api_key" not in serialized
    assert "gsk_" not in serialized


def test_phase_e_values_were_recomputed_from_paired_rows() -> None:
    audit = read_json("PHASE_E_RESULT_AUDIT.json")
    assert audit["paired_n"] == 63
    assert audit["v2_evidence_recall"] == pytest.approx(0.7380952380952381)
    assert audit["v1_evidence_recall"] == pytest.approx(0.746031746031746)
    assert audit["v2_mean_actual_context_tokens"] == pytest.approx(1199.2063492063492)
    assert audit["v1_mean_actual_context_tokens"] == pytest.approx(1147.6190476190477)
    assert audit["v2_point_dominated_by_v1"] is True
    assert audit["frozen_v1_sha256"] == "5c570ec2b4c52e1b0ea0f2152a6f8fb6f1fa1c1575e58cfb58bde2c434a54d05"


def test_smoke_manifest_and_requests_are_exact_and_label_free() -> None:
    manifest = read_json("SMOKE_TEST_MANIFEST.json")
    request_manifest = read_json("SMOKE_TEST_REQUESTS_MANIFEST.json")
    requests = read_jsonl("SMOKE_TEST_REQUESTS.jsonl")
    assert manifest["question_count"] == 16
    assert manifest["dataset_counts"] == {"musique_full": 8, "qasper": 8}
    assert manifest["answerability_counts"] == {"ANSWERABLE": 8, "INSUFFICIENT": 8}
    assert len(requests) == 48
    assert len({(row["question_id"], row["system"]) for row in requests}) == 48
    assert {row["system"] for row in requests} == set(manifest["systems"])
    assert request_manifest["requests_sha256"] == file_sha256(OUT / "SMOKE_TEST_REQUESTS.jsonl")
    for row in requests:
        assert "gold_answers" not in row and "gold_answerability" not in row
        assert [item["evidence_id"] for item in row["evidence"]] == [
            f"E{i}" for i in range(1, len(row["evidence"]) + 1)
        ]


def test_checkpoint_call_count_resume_and_request_mismatch(tmp_path: Path) -> None:
    request = tiny_request()
    fake = DeterministicFakeAnswerer()
    runner = CheckpointRunner(tmp_path, "run-fingerprint", generation(), read_json("FINAL_EVALUATION_PROTOCOL.json")["pricing"])
    first = runner.run([request], fake)
    second = runner.run([request], fake)
    assert first == {"requested": 1, "new_calls": 1, "cache_hits": 0, "checkpoint_records": 1}
    assert second == {"requested": 1, "new_calls": 0, "cache_hits": 1, "checkpoint_records": 1}
    assert fake.calls == 1
    changed = json.loads(json.dumps(request))
    changed["evidence"][0]["text"] = "Changed evidence."
    with pytest.raises(ValueError, match="request fingerprint mismatch"):
        runner.run([changed], fake)


def test_checkpoint_configuration_mismatch_is_rejected(tmp_path: Path) -> None:
    pricing = read_json("FINAL_EVALUATION_PROTOCOL.json")["pricing"]
    CheckpointRunner(tmp_path, "run-fingerprint", generation(), pricing)
    changed = generation()
    changed["decoding"]["max_output_tokens"] = 801
    with pytest.raises(ValueError, match="configuration fingerprint mismatch"):
        CheckpointRunner(tmp_path, "run-fingerprint", changed, pricing)


def test_system_identity_prevents_cross_system_call_sharing() -> None:
    left = tiny_request("sar_rag_v1")
    right = tiny_request("adaptive_single_view_v2")
    assert left["evidence"] == right["evidence"]
    assert request_fingerprint(left, generation()) != request_fingerprint(right, generation())


def test_failed_request_is_logged_and_not_an_empty_answer(tmp_path: Path) -> None:
    class FailingAnswerer:
        def answer(self, request, generation):
            raise RuntimeError("intentional transport failure")

    runner = CheckpointRunner(
        tmp_path, "run-fingerprint", generation(),
        read_json("FINAL_EVALUATION_PROTOCOL.json")["pricing"],
    )
    with pytest.raises(RuntimeError, match="intentional transport failure"):
        runner.run([tiny_request()], FailingAnswerer())
    assert not (tmp_path / "ANSWERS.jsonl").exists()
    failure = json.loads((tmp_path / "FAILURES.jsonl").read_text(encoding="utf-8"))
    assert failure["error_type"] == "RuntimeError"


def test_answerability_parsing_and_metric_ingestion() -> None:
    sufficient = validate_answer({
        "answerability": "ANSWERABLE",
        "claims": [{"text": "Alpha is supported.", "evidence_ids": ["E1"]}],
        "missing_information": "",
    }, 1)
    insufficient = validate_answer({
        "answerability": "INSUFFICIENT", "claims": [],
        "missing_information": "Required evidence is absent.",
    }, 0)
    assert sufficient["rendered_answer"].endswith("[E1]")
    assert insufficient["rendered_answer"] == "Required evidence is absent."
    assert token_f1("The Alpha", ["alpha"]) == pytest.approx(2 / 3)


def test_paid_command_is_blocked_before_client_creation_without_deliberate_flag(tmp_path: Path) -> None:
    result = subprocess.run([
        sys.executable, str(ROOT / "tools/run_final_answer_evaluation.py"),
        "--mode", "paid-smoke",
        "--protocol", str(OUT / "FINAL_EVALUATION_PROTOCOL.json"),
        "--manifest", str(OUT / "SMOKE_TEST_MANIFEST.json"),
        "--checkpoint-dir", str(tmp_path / "paid"),
    ], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert "PAID EXECUTION BLOCKED" in result.stderr
    assert not (tmp_path / "paid").exists()
