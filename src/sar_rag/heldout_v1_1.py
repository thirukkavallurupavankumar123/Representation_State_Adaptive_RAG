"""Strict checkpoint runner for the frozen held-out v1.1 evaluation."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from time import sleep
from datetime import datetime, timedelta, timezone
from typing import Protocol

from .final_evaluation import file_sha256, logical_key, object_sha256, validate_answer
from .generation_v1_1 import request_fingerprint_v1_1
from .smoke_recovery import AttemptOutcome, load_jsonl


class HeldoutAnswerer(Protocol):
    def attempt(self, request: dict) -> AttemptOutcome: ...


def heldout_request_fingerprint_v1_1(request: dict) -> str:
    """Bind the smoke-v1.1 fingerprint to the exact held-out question text."""
    return object_sha256({
        "base_v1_1_request_fingerprint": request_fingerprint_v1_1(request),
        "question": request["question"],
    })


class HeldoutV11Runner:
    """Round-robin retries with immutable protocol, pair, and request manifests."""

    def __init__(self, directory: Path, protocol: dict, manifest: dict,
                 request_manifest: dict, *, protocol_path: Path | None = None,
                 manifest_path: Path | None = None,
                 request_manifest_path: Path | None = None,
                 free_quota_plan: dict | None = None,
                 now_fn=None) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.attempts_path = directory / "API_ATTEMPTS.jsonl"
        self.answers_path = directory / "ANSWERS.jsonl"
        self.failures_path = directory / "FAILURES.jsonl"
        self.meta_path = directory / "CHECKPOINT_META.json"
        self.protocol = deepcopy(protocol)
        self.manifest = deepcopy(manifest)
        self.request_manifest = deepcopy(request_manifest)
        self.free_quota_plan = deepcopy(free_quota_plan)
        self.now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        self.quota_state_path = directory / "FREE_QUOTA_STATE.json"
        self.paths = (protocol_path, manifest_path, request_manifest_path)
        self.file_hashes = tuple(file_sha256(path) if path else None for path in self.paths)
        self.file_stats = tuple(
            (path.stat().st_size, path.stat().st_mtime_ns) if path else None
            for path in self.paths
        )
        if object_sha256(self._body(self.protocol, "protocol_fingerprint")) != self.protocol["protocol_fingerprint"]:
            raise ValueError("invalid held-out protocol fingerprint")
        if object_sha256(self._body(self.manifest, "manifest_fingerprint")) != self.manifest["manifest_fingerprint"]:
            raise ValueError("invalid held-out manifest fingerprint")
        if object_sha256(self._body(self.request_manifest, "request_manifest_fingerprint")) != self.request_manifest["request_manifest_fingerprint"]:
            raise ValueError("invalid held-out request manifest fingerprint")
        self.pinned_protocol_fingerprint = self.protocol["protocol_fingerprint"]
        self.pinned_manifest_fingerprint = self.manifest["manifest_fingerprint"]
        self.pinned_request_manifest_fingerprint = self.request_manifest["request_manifest_fingerprint"]
        expected_meta = {
            "protocol_fingerprint": protocol["protocol_fingerprint"],
            "run_manifest_fingerprint": manifest["manifest_fingerprint"],
            "request_manifest_fingerprint": request_manifest["request_manifest_fingerprint"],
            "generation": protocol["generation"],
            "pair_count": manifest["pair_count"],
        }
        if self.meta_path.exists():
            if json.loads(self.meta_path.read_text(encoding="utf-8")) != expected_meta:
                raise ValueError("held-out checkpoint configuration fingerprint mismatch")
        else:
            self.meta_path.write_text(json.dumps(expected_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.attempts = load_jsonl(self.attempts_path)
        self.answers = {row["logical_key"]: row for row in load_jsonl(self.answers_path)}
        self.pairs = {row["logical_key"]: row for row in manifest["pairs"]}
        self.request_pairs = {row["logical_key"]: row for row in request_manifest["pairs"]}

    @staticmethod
    def _parse_time(value: str) -> datetime:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    def _quota_preflight(self, reserved_tokens: int) -> dict | None:
        if not self.free_quota_plan:
            return None
        now = self.now_fn()
        if self.quota_state_path.exists():
            saved = json.loads(self.quota_state_path.read_text(encoding="utf-8"))
            resume_value = saved.get("resume_not_before_utc")
            if saved.get("status") == "PAUSED_FREE_QUOTA" and resume_value and now < self._parse_time(resume_value):
                return {**saved, "reason": saved.get("reason", "saved_quota_pause")}
        limits = self.free_quota_plan["limits"]
        events = []
        for row in self.attempts:
            if not row.get("started_at_utc"):
                continue
            when = self._parse_time(row["started_at_utc"])
            tokens = int(row.get("input_tokens", 0)) + int(row.get("output_tokens", 0))
            if tokens <= 0:
                tokens = int(row.get("quota_reserved_tokens", 0))
            events.append((when, tokens))
        checks = (
            ("minute", timedelta(seconds=60), int(limits["rpm"]), int(limits["tpm"])),
            ("day", timedelta(hours=24), int(limits["rpd"]), int(limits["tpd"])),
        )
        for name, window, request_limit, token_limit in checks:
            active = [(when, tokens) for when, tokens in events if when > now - window]
            if len(active) + 1 > request_limit or sum(tokens for _, tokens in active) + reserved_tokens > token_limit:
                oldest = min((when for when, _ in active), default=now)
                return {
                    "status": "PAUSED_FREE_QUOTA",
                    "reason": f"free_{name}_quota_reservation",
                    "paused_at_utc": now.isoformat(),
                    "resume_not_before_utc": (oldest + window + timedelta(seconds=1)).isoformat(),
                    "active_requests": len(active),
                    "active_tokens": sum(tokens for _, tokens in active),
                    "next_reserved_tokens": reserved_tokens,
                }
        return None

    def _write_quota_pause(self, state: dict) -> None:
        state = {**state,
                 "quota_plan_fingerprint": self.free_quota_plan.get("quota_plan_fingerprint") if self.free_quota_plan else None,
                 "checkpoint_attempts": len(self.attempts), "checkpoint_answers": len(self.answers)}
        self.quota_state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @staticmethod
    def _body(value: dict, fingerprint_field: str) -> dict:
        return {key: item for key, item in value.items() if key != fingerprint_field}

    def _verify_frozen(self) -> None:
        if self.protocol["protocol_fingerprint"] != self.pinned_protocol_fingerprint:
            raise RuntimeError("held-out protocol fingerprint mismatch before call")
        # The large pair maps are deep-copied after an initialization-time full
        # hash.  Every call checks their declared digests and backing-file
        # identities; a changed file is then fully rehashed and rejected.
        if self.manifest["manifest_fingerprint"] != self.pinned_manifest_fingerprint:
            raise RuntimeError("held-out manifest fingerprint mismatch before call")
        if self.request_manifest["request_manifest_fingerprint"] != self.pinned_request_manifest_fingerprint:
            raise RuntimeError("held-out request manifest fingerprint mismatch before call")
        if self.manifest["protocol_fingerprint"] != self.protocol["protocol_fingerprint"]:
            raise RuntimeError("held-out protocol/manifest linkage mismatch before call")
        if self.request_manifest["protocol_fingerprint"] != self.protocol["protocol_fingerprint"] or self.request_manifest["run_manifest_fingerprint"] != self.manifest["manifest_fingerprint"]:
            raise RuntimeError("held-out request-manifest linkage mismatch before call")
        for path, expected_hash, expected_stat in zip(self.paths, self.file_hashes, self.file_stats):
            if path:
                current_stat = (path.stat().st_size, path.stat().st_mtime_ns)
                if current_stat != expected_stat:
                    raise RuntimeError(f"held-out frozen file changed before call: {path.name}")

    @staticmethod
    def _append(path: Path, row: dict) -> None:
        payload = json.dumps(row, sort_keys=True) + "\n"
        for attempt in range(20):
            try:
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(payload)
                    stream.flush()
                return
            except PermissionError:
                if attempt == 19:
                    raise
                sleep(0.05 * (attempt + 1))

    def run(self, requests: list[dict], answerer: HeldoutAnswerer) -> dict:
        keys = [logical_key(row) for row in requests]
        if len(keys) != self.manifest["pair_count"] or len(set(keys)) != len(keys) or set(keys) != set(self.pairs):
            raise ValueError("requests do not match the exact frozen held-out pair set")
        computed = {logical_key(row): heldout_request_fingerprint_v1_1(row) for row in requests}
        declared = {key: row["heldout_request_fingerprint_v1_1"] for key, row in self.request_pairs.items()}
        if computed != declared:
            raise ValueError("request fingerprints do not match the frozen request manifest")
        attempts_by_key = Counter(row["logical_key"] for row in self.attempts)
        max_total = self.manifest["maximum_total_api_attempts_including_retries"]
        max_per_pair = self.manifest["maximum_attempts_per_pair_including_first"]
        # A sealed request manifest may tighten (never expand) the earlier
        # protocol-level ceiling after exact input reservations are known.
        max_cost = min(
            float(self.manifest["maximum_cost_usd"]),
            float(self.request_manifest.get(
                "maximum_cost_usd_from_sealed_requests",
                self.manifest["maximum_cost_usd"],
            )),
        )
        pricing = self.protocol["cost_and_attempt_plan"]["pricing"]
        max_output = self.protocol["generation"]["maximum_completion_tokens"]
        new_attempts = new_answers = 0
        reserved_total = sum(float(row["authorized_worst_case_cost_usd"]) for row in self.attempts)
        for round_number in range(1, max_per_pair + 1):
            for request in requests:
                key = logical_key(request)
                if key in self.answers or attempts_by_key[key] >= round_number:
                    continue
                if len(self.attempts) >= max_total:
                    return self.summary(new_attempts, new_answers, "global_attempt_ceiling")
                reserved = reserved_total
                input_ceiling = int(self.request_pairs[key]["conservative_billed_input_tokens"])
                attempt_ceiling = (
                    input_ceiling * pricing["input_per_million"]
                    + max_output * pricing["output_per_million"]
                ) / 1_000_000
                quota_reserved_tokens = input_ceiling + max_output
                quota_pause = self._quota_preflight(quota_reserved_tokens)
                if quota_pause:
                    self._write_quota_pause(quota_pause)
                    return self.summary(new_attempts, new_answers, quota_pause["reason"])
                if reserved + attempt_ceiling > max_cost + 1e-12:
                    return self.summary(new_attempts, new_answers, "global_cost_ceiling")
                self._verify_frozen()
                outcome = answerer.attempt(request)
                started_at = self.now_fn().isoformat()
                attempt_number = attempts_by_key[key] + 1
                attempt = {
                    "logical_key": key,
                    "request_fingerprint": computed[key],
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
                    "started_at_utc": started_at,
                    "quota_reserved_tokens": quota_reserved_tokens,
                    "http_status": outcome.http_status,
                    "rate_limit_headers": outcome.rate_limit_headers,
                }
                validated = None
                if outcome.payload is not None:
                    try:
                        validated = validate_answer(outcome.payload, len(request["evidence"]))
                        attempt["response_valid"] = True
                    except Exception as exc:
                        attempt["validation_error_type"] = type(exc).__name__
                        attempt["validation_error"] = str(exc)
                self._append(self.attempts_path, attempt)
                self.attempts.append(attempt)
                reserved_total += attempt_ceiling
                attempts_by_key[key] += 1
                new_attempts += 1
                if outcome.http_status == 429:
                    headers = outcome.rate_limit_headers or {}
                    delay = float(headers.get("retry-after", 60)) if str(headers.get("retry-after", "60")).replace(".", "", 1).isdigit() else 60.0
                    pause = {
                        "status": "PAUSED_FREE_QUOTA", "reason": "provider_http_429",
                        "paused_at_utc": self.now_fn().isoformat(),
                        "resume_not_before_utc": (self.now_fn() + timedelta(seconds=delay + 1)).isoformat(),
                        "provider_headers": headers,
                    }
                    self._write_quota_pause(pause)
                    return self.summary(new_attempts, new_answers, "provider_http_429")
                if outcome.status == "fatal_error":
                    self._append(self.failures_path, attempt)
                    raise RuntimeError(f"fatal held-out request failure: {outcome.error}")
                if validated is None:
                    failure = dict(attempt)
                    if outcome.raw_response is not None:
                        failure["raw_response"] = outcome.raw_response
                    self._append(self.failures_path, failure)
                    continue
                cost = (
                    outcome.input_tokens * pricing["input_per_million"]
                    + outcome.output_tokens * pricing["output_per_million"]
                ) / 1_000_000
                answer = {
                    "logical_key": key,
                    "request_fingerprint": computed[key],
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
        counts = Counter(row["logical_key"] for row in self.attempts)
        return {
            "new_attempts": new_attempts,
            "new_answers": new_answers,
            "cumulative_attempts": len(self.attempts),
            "first_attempt_valid_pairs": sum(row["attempt_for_pair"] == 1 for row in self.answers.values()),
            "eventually_valid_pairs": len(self.answers),
            "persistent_failure_pairs": sorted(set(self.pairs) - set(self.answers)),
            "attempts_by_pair": dict(sorted(counts.items())),
            "stop_reason": stop_reason,
        }
