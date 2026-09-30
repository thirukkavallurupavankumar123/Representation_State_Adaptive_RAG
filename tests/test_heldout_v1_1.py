from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from sar_rag.final_evaluation import logical_key, object_sha256
from sar_rag.generation_v1_1 import generation_config_v1_1
from sar_rag.heldout_v1_1 import HeldoutV11Runner, heldout_request_fingerprint_v1_1
from sar_rag.smoke_recovery import AttemptOutcome, load_jsonl


def valid() -> AttemptOutcome:
    payload = {"answerability": "ANSWERABLE", "claims": [{"text": "fact", "evidence_ids": ["E1"]}], "missing_information": ""}
    return AttemptOutcome(status="success", payload=payload, raw_response=json.dumps(payload), input_tokens=100, output_tokens=10, latency_ms=1)


class Scripted:
    def __init__(self, values: dict[str, list[AttemptOutcome]]) -> None:
        self.values = {key: deque(items) for key, items in values.items()}
        self.calls: list[str] = []

    def attempt(self, request: dict) -> AttemptOutcome:
        key = logical_key(request)
        self.calls.append(key)
        return self.values[key].popleft()


def fixture() -> tuple[list[dict], dict, dict, dict]:
    requests = [{"dataset": "d", "question_id": f"q{i}", "system": "s", "question": "q?", "evidence": [{"evidence_id": "E1", "text": "e"}]} for i in range(2)]
    protocol = {
        "generation": {**generation_config_v1_1(), "maximum_completion_tokens": 1600},
        "cost_and_attempt_plan": {"pricing": {"input_per_million": 0.15, "output_per_million": 0.6}},
    }
    protocol["protocol_fingerprint"] = object_sha256(protocol)
    pairs = [{"logical_key": logical_key(row)} for row in requests]
    manifest = {
        "protocol_fingerprint": protocol["protocol_fingerprint"], "pair_count": 2,
        "maximum_total_api_attempts_including_retries": 6,
        "maximum_attempts_per_pair_including_first": 3,
        "maximum_cost_usd": 1.0, "pairs": pairs,
    }
    manifest["manifest_fingerprint"] = object_sha256(manifest)
    request_manifest = {
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "run_manifest_fingerprint": manifest["manifest_fingerprint"], "pairs": [{
            "logical_key": logical_key(row), "heldout_request_fingerprint_v1_1": heldout_request_fingerprint_v1_1(row),
            "conservative_billed_input_tokens": 2800,
        } for row in requests],
    }
    request_manifest["request_manifest_fingerprint"] = object_sha256(request_manifest)
    return requests, protocol, manifest, request_manifest


def test_round_robin_retry_and_exact_resume(tmp_path: Path) -> None:
    requests, protocol, manifest, request_manifest = fixture()
    failed = logical_key(requests[0])
    answerer = Scripted({failed: [AttemptOutcome(status="api_error", error="429"), valid()], logical_key(requests[1]): [valid()]})
    first = HeldoutV11Runner(tmp_path, protocol, manifest, request_manifest).run(requests, answerer)
    assert first["cumulative_attempts"] == 3
    assert first["first_attempt_valid_pairs"] == 1
    assert first["eventually_valid_pairs"] == 2
    resume_answerer = Scripted({key: [] for key in (failed, logical_key(requests[1]))})
    resume = HeldoutV11Runner(tmp_path, protocol, manifest, request_manifest).run(requests, resume_answerer)
    assert resume["new_attempts"] == 0
    assert resume_answerer.calls == []


def test_persistent_failure_retained(tmp_path: Path) -> None:
    requests, protocol, manifest, request_manifest = fixture()
    failed = logical_key(requests[0])
    answerer = Scripted({failed: [AttemptOutcome(status="api_error", error="x") for _ in range(3)], logical_key(requests[1]): [valid()]})
    result = HeldoutV11Runner(tmp_path, protocol, manifest, request_manifest).run(requests, answerer)
    assert result["persistent_failure_pairs"] == [failed]
    assert len(load_jsonl(tmp_path / "API_ATTEMPTS.jsonl")) == 4


def test_request_drift_rejected_before_call(tmp_path: Path) -> None:
    requests, protocol, manifest, request_manifest = fixture()
    requests[0]["question"] = "changed"
    answerer = Scripted({logical_key(row): [valid()] for row in requests})
    with pytest.raises(ValueError, match="request fingerprints"):
        HeldoutV11Runner(tmp_path, protocol, manifest, request_manifest).run(requests, answerer)
    assert answerer.calls == []


def test_free_quota_pause_and_resume_without_duplicate_calls(tmp_path: Path) -> None:
    requests, protocol, manifest, request_manifest = fixture()
    clock = [datetime(2026, 9, 27, tzinfo=timezone.utc)]
    quota = {"limits": {"rpm": 30, "rpd": 1000, "tpm": 8000, "tpd": 4500}, "quota_plan_fingerprint": "test"}
    answerer = Scripted({logical_key(row): [valid()] for row in requests})
    first = HeldoutV11Runner(tmp_path, protocol, manifest, request_manifest,
                             free_quota_plan=quota, now_fn=lambda: clock[0]).run(requests, answerer)
    assert first["eventually_valid_pairs"] == 1
    assert first["stop_reason"] == "free_day_quota_reservation"
    assert len(answerer.calls) == 1
    paused = HeldoutV11Runner(tmp_path, protocol, manifest, request_manifest,
                              free_quota_plan=quota, now_fn=lambda: clock[0]).run(requests, Scripted({logical_key(row): [valid()] for row in requests}))
    assert paused["new_attempts"] == 0
    clock[0] += timedelta(hours=24, seconds=2)
    resumed_answerer = Scripted({logical_key(row): [valid()] for row in requests})
    resumed = HeldoutV11Runner(tmp_path, protocol, manifest, request_manifest,
                               free_quota_plan=quota, now_fn=lambda: clock[0]).run(requests, resumed_answerer)
    assert resumed["eventually_valid_pairs"] == 2
    assert resumed_answerer.calls == [logical_key(requests[1])]


def test_http_429_pauses_immediately(tmp_path: Path) -> None:
    requests, protocol, manifest, request_manifest = fixture()
    clock = [datetime(2026, 9, 27, tzinfo=timezone.utc)]
    quota = {"limits": {"rpm": 30, "rpd": 1000, "tpm": 8000, "tpd": 200000}, "quota_plan_fingerprint": "test"}
    first_key = logical_key(requests[0])
    answerer = Scripted({
        first_key: [AttemptOutcome(status="api_error", error="Groq HTTP 429", http_status=429, rate_limit_headers={"retry-after": "120"})],
        logical_key(requests[1]): [valid()],
    })
    result = HeldoutV11Runner(tmp_path, protocol, manifest, request_manifest,
                              free_quota_plan=quota, now_fn=lambda: clock[0]).run(requests, answerer)
    assert result["stop_reason"] == "provider_http_429"
    assert answerer.calls == [first_key]
    state = json.loads((tmp_path / "FREE_QUOTA_STATE.json").read_text(encoding="utf-8"))
    assert state["resume_not_before_utc"] == (clock[0] + timedelta(seconds=121)).isoformat()
