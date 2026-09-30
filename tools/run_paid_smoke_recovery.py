"""Run the separately authorized, protocol-preserving paid-smoke recovery."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter, sleep


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sar_rag.final_evaluation import ANSWER_SCHEMA, build_messages, file_sha256, logical_key  # noqa: E402
from sar_rag.smoke_recovery import AttemptOutcome, RecoveryRunner, load_jsonl  # noqa: E402


def sanitize_error(value: str) -> str:
    value = re.sub(r"organization `[^`]+`", "organization `[redacted]`", value)
    value = re.sub(r"gsk_[A-Za-z0-9]+", "[redacted-api-key]", value)
    return value[:2000]


def retry_after_seconds(value: str) -> float | None:
    match = re.search(r"try again in\s+([0-9.]+)\s*(ms|s)", value, re.IGNORECASE)
    if not match:
        return None
    delay = float(match.group(1))
    return delay / 1000 if match.group(2).lower() == "ms" else delay


class GroqRecoveryAnswerer:
    def __init__(self, client, minimum_interval_seconds: float) -> None:
        self.client = client
        self.minimum_interval_seconds = minimum_interval_seconds
        self.last_started: float | None = None
        self.not_before: float = 0.0

    def _wait_for_slot(self) -> None:
        while True:
            interval_deadline = (
                self.last_started + self.minimum_interval_seconds
                if self.last_started is not None else 0.0
            )
            delay = max(interval_deadline, self.not_before) - perf_counter()
            if delay <= 0:
                return
            sleep(min(delay, 30.0))

    def attempt(self, request: dict, generation: dict) -> AttemptOutcome:
        self._wait_for_slot()
        self.last_started = perf_counter()
        messages, _ = build_messages(request["question"], request["evidence"])
        usage_before = len(self.client.usage_log)
        started = perf_counter()
        try:
            raw = self.client.complete(
                messages,
                "frozen_final_answer",
                ANSWER_SCHEMA,
                int(generation["decoding"]["max_output_tokens"]),
            )
            payload = json.loads(raw)
        except Exception as exc:
            error = sanitize_error(str(exc))
            retry_delay = retry_after_seconds(error)
            if retry_delay is not None:
                self.not_before = max(self.not_before, perf_counter() + retry_delay + 0.25)
            return AttemptOutcome(
                status="api_error",
                latency_ms=(perf_counter() - started) * 1000,
                error_type=type(exc).__name__,
                error=error,
            )
        if len(self.client.usage_log) != usage_before + 1:
            return AttemptOutcome(
                status="usage_accounting_error",
                raw_response=raw,
                latency_ms=(perf_counter() - started) * 1000,
                error_type="RuntimeError",
                error="successful response lacks exactly one usage record",
            )
        usage = self.client.usage_log[-1]
        return AttemptOutcome(
            status="success",
            payload=payload,
            raw_response=raw,
            input_tokens=int(usage["prompt_tokens"]),
            output_tokens=int(usage["completion_tokens"]),
            latency_ms=float(usage["latency_seconds"]) * 1000,
        )


def verify_authorization(path: Path | None, manifest: dict) -> None:
    if path is None or not path.is_file():
        raise SystemExit("PAID RECOVERY BLOCKED: a new authorization file is absent")
    actual = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "EXPLICIT_USER_AUTHORIZATION_RECORDED",
        "authorized_stage": "paid_smoke_recovery_v1_only",
        "user_authorized": True,
        "recovery_manifest_fingerprint": manifest["manifest_fingerprint"],
        "maximum_total_api_attempts_including_retries": manifest[
            "maximum_total_api_attempts_including_retries"
        ],
        "maximum_retries_per_pair": manifest["maximum_retries_per_pair"],
        "maximum_cost_usd": manifest["maximum_cost_usd"],
    }
    if actual != expected:
        raise SystemExit("PAID RECOVERY BLOCKED: authorization content/fingerprint mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--authorization-file", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--execute-paid", action="store_true")
    args = parser.parse_args()
    if not args.execute_paid:
        raise SystemExit("PAID RECOVERY BLOCKED: --execute-paid was not provided")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    verify_authorization(args.authorization_file, manifest)
    output = ROOT / "output/final_evaluation"
    for name, expected_hash in manifest["original_integrity_sha256"].items():
        if file_sha256(output / name) != expected_hash:
            raise SystemExit(f"PAID RECOVERY BLOCKED: original evidence changed: {name}")
    if args.checkpoint_dir.resolve() == (output / "paid_smoke_checkpoint").resolve():
        raise SystemExit("PAID RECOVERY BLOCKED: original checkpoint cannot be the recovery target")

    requests = load_jsonl(output / "SMOKE_TEST_REQUESTS.jsonl")
    pair_keys = {row["logical_key"] for row in manifest["pairs"]}
    recovery_requests = [row for row in requests if logical_key(row) in pair_keys]
    protocol = json.loads((output / "FINAL_EVALUATION_PROTOCOL.json").read_text(encoding="utf-8"))
    checkpoint_meta = json.loads(
        (output / "paid_smoke_checkpoint/CHECKPOINT_META.json").read_text(encoding="utf-8")
    )
    generation = checkpoint_meta["generation"]
    from sar_rag.environment import load_project_env
    load_project_env()
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit("PAID RECOVERY BLOCKED: GROQ_API_KEY is unavailable")
    from sar_rag.groq import GroqClient
    client = GroqClient(
        generation["model"],
        timeout_seconds=protocol["generation"]["timeout_seconds"],
        max_retries=0,
    )
    runner = RecoveryRunner(
        args.checkpoint_dir,
        manifest["manifest_fingerprint"],
        generation,
        protocol["pricing"],
        pair_keys,
        manifest["maximum_total_api_attempts_including_retries"],
        manifest["maximum_attempts_per_pair_including_initial"],
        manifest["maximum_cost_usd"],
        {row["logical_key"]: row["preflight_input_tokens"] for row in manifest["pairs"]},
    )
    answerer = GroqRecoveryAnswerer(
        client, float(manifest["minimum_seconds_between_attempt_starts"])
    )
    summary = runner.run(recovery_requests, answerer)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
