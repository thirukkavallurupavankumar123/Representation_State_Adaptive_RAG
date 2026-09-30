"""Protocol-preserving recovery plumbing for missing paid-smoke answers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Protocol

from .final_evaluation import (
    AnswerResponse, logical_key, object_sha256, request_fingerprint, validate_answer,
)


RECOVERY_VERSION = "sar-rag-paid-smoke-recovery-1.0.0"


@dataclass(frozen=True)
class AttemptOutcome:
    """One external attempt or a simulated equivalent."""

    status: str
    payload: dict | None = None
    raw_response: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    error_type: str | None = None
    error: str | None = None
    http_status: int | None = None
    rate_limit_headers: dict[str, str] | None = None


class RecoveryAnswerer(Protocol):
    def attempt(self, request: dict, generation: dict) -> AttemptOutcome: ...


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def missing_requests(requests: list[dict], saved_answers: list[dict]) -> list[dict]:
    answered = {row["logical_key"] for row in saved_answers}
    return [row for row in requests if logical_key(row) not in answered]


class RecoveryRunner:
    """Round-robin retries with a persistent global and per-pair attempt ceiling."""

    def __init__(
        self,
        directory: Path,
        run_fingerprint: str,
        generation: dict,
        pricing: dict,
        allowed_keys: set[str],
        maximum_total_attempts: int,
        maximum_attempts_per_pair: int,
        maximum_cost_usd: float,
        preflight_input_tokens_by_key: dict[str, int],
    ) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.attempts_path = directory / "API_ATTEMPTS.jsonl"
        self.answers_path = directory / "ANSWERS.jsonl"
        self.failures_path = directory / "FAILURES.jsonl"
        self.meta_path = directory / "CHECKPOINT_META.json"
        self.maximum_total_attempts = maximum_total_attempts
        self.maximum_attempts_per_pair = maximum_attempts_per_pair
        self.maximum_cost_usd = maximum_cost_usd
        self.preflight_input_tokens_by_key = preflight_input_tokens_by_key
        self.allowed_keys = allowed_keys
        self.generation = generation
        self.pricing = pricing
        expected_meta = {
            "recovery_version": RECOVERY_VERSION,
            "run_fingerprint": run_fingerprint,
            "generation_fingerprint": object_sha256(generation),
            "generation": generation,
            "allowed_keys": sorted(allowed_keys),
            "maximum_total_attempts": maximum_total_attempts,
            "maximum_attempts_per_pair": maximum_attempts_per_pair,
            "maximum_cost_usd": maximum_cost_usd,
            "preflight_input_tokens_by_key": preflight_input_tokens_by_key,
        }
        if self.meta_path.exists():
            if json.loads(self.meta_path.read_text(encoding="utf-8")) != expected_meta:
                raise ValueError("recovery checkpoint configuration fingerprint mismatch")
        else:
            self.meta_path.write_text(
                json.dumps(expected_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        self.attempts = load_jsonl(self.attempts_path)
        self.records = {row["logical_key"]: row for row in load_jsonl(self.answers_path)}

    def _append(self, path: Path, row: dict) -> None:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    def run(self, requests: list[dict], answerer: RecoveryAnswerer) -> dict:
        request_keys = {logical_key(row) for row in requests}
        if request_keys != self.allowed_keys or len(requests) != len(request_keys):
            raise ValueError("recovery request set differs from authorized missing-pair set")
        attempts_by_key = Counter(row["logical_key"] for row in self.attempts)
        new_attempts = new_answers = 0
        for round_number in range(1, self.maximum_attempts_per_pair + 1):
            for request in requests:
                key = logical_key(request)
                if key in self.records or attempts_by_key[key] >= round_number:
                    continue
                if len(self.attempts) >= self.maximum_total_attempts:
                    return self.summary(new_attempts, new_answers, exhausted=True, cost_exhausted=False)
                attempt_cost_ceiling = (
                    self.preflight_input_tokens_by_key[key] * self.pricing["input_per_million"]
                    + self.generation["decoding"]["max_output_tokens"]
                    * self.pricing["output_per_million"]
                ) / 1_000_000
                reserved_cost = sum(
                    float(row["authorized_worst_case_cost_usd"]) for row in self.attempts
                )
                if reserved_cost + attempt_cost_ceiling > self.maximum_cost_usd + 1e-12:
                    return self.summary(new_attempts, new_answers, exhausted=False, cost_exhausted=True)
                started = perf_counter()
                try:
                    outcome = answerer.attempt(request, self.generation)
                except Exception as exc:  # transport adapters should normally return an outcome
                    outcome = AttemptOutcome(
                        status="adapter_error", error_type=type(exc).__name__, error=str(exc)
                    )
                attempt_number = attempts_by_key[key] + 1
                attempt_row = {
                    "logical_key": key,
                    "request_fingerprint": request_fingerprint(request, self.generation),
                    "attempt_for_pair": attempt_number,
                    "recovery_round": round_number,
                    "status": outcome.status,
                    "error_type": outcome.error_type,
                    "error": outcome.error,
                    "input_tokens": outcome.input_tokens,
                    "output_tokens": outcome.output_tokens,
                    "latency_ms": outcome.latency_ms,
                    "runner_wall_ms": (perf_counter() - started) * 1000,
                    "authorized_worst_case_cost_usd": attempt_cost_ceiling,
                    "remaining_attempt_allowance_before_call": self.maximum_total_attempts - len(self.attempts),
                    "remaining_cost_allowance_before_call_usd": self.maximum_cost_usd - reserved_cost,
                }
                self._append(self.attempts_path, attempt_row)
                self.attempts.append(attempt_row)
                attempts_by_key[key] += 1
                new_attempts += 1

                if outcome.payload is None:
                    self._append(self.failures_path, attempt_row)
                    continue
                try:
                    validated = validate_answer(outcome.payload, len(request["evidence"]))
                except Exception as exc:
                    failure = {
                        **attempt_row,
                        "status": "local_validation_error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "raw_response_sha256": object_sha256(outcome.raw_response or outcome.payload),
                        "raw_response": outcome.raw_response,
                    }
                    self._append(self.failures_path, failure)
                    continue

                cost = (
                    outcome.input_tokens * self.pricing["input_per_million"]
                    + outcome.output_tokens * self.pricing["output_per_million"]
                ) / 1_000_000
                answer = {
                    "logical_key": key,
                    "request_fingerprint": request_fingerprint(request, self.generation),
                    "dataset": request["dataset"],
                    "question_id": request["question_id"],
                    "system": request["system"],
                    "response": validated,
                    "input_tokens": outcome.input_tokens,
                    "output_tokens": outcome.output_tokens,
                    "total_tokens": outcome.input_tokens + outcome.output_tokens,
                    "latency_ms": outcome.latency_ms,
                    "attempt_for_pair": attempt_number,
                    "estimated_cost_usd": cost,
                }
                self._append(self.answers_path, answer)
                self.records[key] = answer
                new_answers += 1
        return self.summary(new_attempts, new_answers, exhausted=False, cost_exhausted=False)

    def summary(self, new_attempts: int, new_answers: int, exhausted: bool,
                cost_exhausted: bool) -> dict:
        reserved_cost = sum(
            float(row["authorized_worst_case_cost_usd"]) for row in self.attempts
        )
        return {
            "new_attempts": new_attempts,
            "new_answers": new_answers,
            "cumulative_attempts": len(self.attempts),
            "cumulative_answers": len(self.records),
            "missing_answers": len(self.allowed_keys - set(self.records)),
            "attempt_ceiling": self.maximum_total_attempts,
            "attempt_ceiling_exhausted": exhausted or len(self.attempts) >= self.maximum_total_attempts,
            "authorized_cost_ceiling_usd": self.maximum_cost_usd,
            "cumulative_authorized_worst_case_cost_usd": reserved_cost,
            "cost_ceiling_exhausted": cost_exhausted,
        }
