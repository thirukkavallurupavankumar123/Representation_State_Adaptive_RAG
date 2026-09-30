"""Checkpoint runner for the fresh 48-pair smoke-generation v1.1 run."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Protocol

from .final_evaluation import file_sha256, logical_key, object_sha256, validate_answer
from .generation_v1_1 import request_fingerprint_v1_1
from .smoke_recovery import AttemptOutcome, load_jsonl


class V11Answerer(Protocol):
    def attempt(self, request: dict) -> AttemptOutcome: ...


class V11SmokeRunner:
    def __init__(self, directory: Path, protocol: dict, manifest: dict,
                 protocol_path: Path | None = None,
                 manifest_path: Path | None = None) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.attempts_path = directory / "API_ATTEMPTS.jsonl"
        self.answers_path = directory / "ANSWERS.jsonl"
        self.failures_path = directory / "FAILURES.jsonl"
        self.meta_path = directory / "CHECKPOINT_META.json"
        self.protocol = protocol
        self.manifest = manifest
        self.protocol_path = protocol_path
        self.manifest_path = manifest_path
        self.protocol_file_sha256 = file_sha256(protocol_path) if protocol_path else None
        self.manifest_file_sha256 = file_sha256(manifest_path) if manifest_path else None
        expected_meta = {
            "protocol_fingerprint": protocol["protocol_fingerprint"],
            "run_manifest_fingerprint": manifest["manifest_fingerprint"],
            "generation": protocol["generation"],
            "pair_count": 48,
        }
        if self.meta_path.exists():
            if json.loads(self.meta_path.read_text(encoding="utf-8")) != expected_meta:
                raise ValueError("v1.1 checkpoint configuration mismatch")
        else:
            self.meta_path.write_text(
                json.dumps(expected_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        self.attempts = load_jsonl(self.attempts_path)
        self.answers = {row["logical_key"]: row for row in load_jsonl(self.answers_path)}
        self.pair_manifest = {row["logical_key"]: row for row in manifest["pairs"]}

    def _verify_frozen_fingerprints(self) -> None:
        protocol_body = {
            key: value for key, value in self.protocol.items()
            if key != "protocol_fingerprint"
        }
        manifest_body = {
            key: value for key, value in self.manifest.items()
            if key != "manifest_fingerprint"
        }
        if object_sha256(protocol_body) != self.protocol["protocol_fingerprint"]:
            raise RuntimeError("v1.1 protocol fingerprint mismatch before call")
        if object_sha256(manifest_body) != self.manifest["manifest_fingerprint"]:
            raise RuntimeError("v1.1 manifest fingerprint mismatch before call")
        if self.manifest["protocol_fingerprint"] != self.protocol["protocol_fingerprint"]:
            raise RuntimeError("v1.1 protocol/manifest linkage mismatch before call")
        if self.protocol_path and file_sha256(self.protocol_path) != self.protocol_file_sha256:
            raise RuntimeError("v1.1 protocol file changed before call")
        if self.manifest_path and file_sha256(self.manifest_path) != self.manifest_file_sha256:
            raise RuntimeError("v1.1 manifest file changed before call")

    @staticmethod
    def _append(path: Path, row: dict) -> None:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    def run(self, requests: list[dict], answerer: V11Answerer) -> dict:
        keys = [logical_key(row) for row in requests]
        if len(keys) != 48 or len(set(keys)) != 48 or set(keys) != set(self.pair_manifest):
            raise ValueError("v1.1 requires the exact frozen 48-pair request set")
        attempts_by_key = Counter(row["logical_key"] for row in self.attempts)
        max_total = self.manifest["maximum_total_api_attempts_including_retries"]
        max_per_pair = self.manifest["maximum_attempts_per_pair_including_first"]
        max_cost = self.manifest["maximum_cost_usd"]
        pricing = self.protocol["cost"]["pricing"]
        max_output = self.protocol["generation"]["decoding"]["max_output_tokens"]
        new_attempts = new_answers = 0
        for round_number in range(1, max_per_pair + 1):
            for request in requests:
                key = logical_key(request)
                if key in self.answers or attempts_by_key[key] >= round_number:
                    continue
                if len(self.attempts) >= max_total:
                    return self.summary(new_attempts, new_answers, "global_attempt_ceiling")
                reserved = sum(float(row["authorized_worst_case_cost_usd"]) for row in self.attempts)
                input_tokens = self.pair_manifest[key]["conservative_billed_input_tokens"]
                attempt_ceiling = (
                    input_tokens * pricing["input_per_million"]
                    + max_output * pricing["output_per_million"]
                ) / 1_000_000
                if reserved + attempt_ceiling > max_cost + 1e-12:
                    return self.summary(new_attempts, new_answers, "global_cost_ceiling")

                self._verify_frozen_fingerprints()
                outcome = answerer.attempt(request)
                attempt_number = attempts_by_key[key] + 1
                attempt = {
                    "logical_key": key,
                    "request_fingerprint": request_fingerprint_v1_1(request),
                    "attempt_for_pair": attempt_number,
                    "round": round_number,
                    "transport_status": outcome.status,
                    "input_tokens": outcome.input_tokens,
                    "output_tokens": outcome.output_tokens,
                    "latency_ms": outcome.latency_ms,
                    "error_type": outcome.error_type,
                    "error": outcome.error,
                    "remaining_attempt_allowance_before_call": max_total - len(self.attempts),
                    "remaining_cost_allowance_before_call_usd": max_cost - reserved,
                    "authorized_worst_case_cost_usd": attempt_ceiling,
                    "response_valid": False,
                }
                validation_error: Exception | None = None
                validated = None
                if outcome.payload is not None:
                    try:
                        validated = validate_answer(outcome.payload, len(request["evidence"]))
                        attempt["response_valid"] = True
                    except Exception as exc:
                        validation_error = exc
                        attempt["validation_error_type"] = type(exc).__name__
                        attempt["validation_error"] = str(exc)
                self._append(self.attempts_path, attempt)
                self.attempts.append(attempt)
                attempts_by_key[key] += 1
                new_attempts += 1

                if outcome.status == "fatal_error":
                    self._append(self.failures_path, attempt)
                    raise RuntimeError(f"fatal v1.1 request failure: {outcome.error}")
                if validated is None:
                    failure = dict(attempt)
                    if validation_error is not None:
                        failure["raw_response"] = outcome.raw_response
                    self._append(self.failures_path, failure)
                    continue

                cost = (
                    outcome.input_tokens * pricing["input_per_million"]
                    + outcome.output_tokens * pricing["output_per_million"]
                ) / 1_000_000
                answer = {
                    "logical_key": key,
                    "request_fingerprint": request_fingerprint_v1_1(request),
                    "dataset": request["dataset"],
                    "question_id": request["question_id"],
                    "system": request["system"],
                    "response": validated,
                    "input_tokens": outcome.input_tokens,
                    "output_tokens": outcome.output_tokens,
                    "total_tokens": outcome.input_tokens + outcome.output_tokens,
                    "latency_ms": outcome.latency_ms,
                    "attempt_for_pair": attempt_number,
                    "first_attempt_valid": attempt_number == 1,
                    "eventually_valid": True,
                    "estimated_cost_usd": cost,
                }
                self._append(self.answers_path, answer)
                self.answers[key] = answer
                new_answers += 1
        return self.summary(new_attempts, new_answers, "completed_retry_schedule")

    def summary(self, new_attempts: int, new_answers: int, stop_reason: str) -> dict:
        attempt_counts = Counter(row["logical_key"] for row in self.attempts)
        first_valid = sum(1 for row in self.answers.values() if row["attempt_for_pair"] == 1)
        eventual_valid = len(self.answers)
        return {
            "new_attempts": new_attempts,
            "new_answers": new_answers,
            "cumulative_attempts": len(self.attempts),
            "first_attempt_valid_pairs": first_valid,
            "eventually_valid_pairs": eventual_valid,
            "persistent_failure_pairs": sorted(set(self.pair_manifest) - set(self.answers)),
            "attempts_by_pair": dict(sorted(attempt_counts.items())),
            "stop_reason": stop_reason,
        }
