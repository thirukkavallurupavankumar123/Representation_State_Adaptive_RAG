"""Run the deterministic dry run or a separately authorized paid smoke test."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter, sleep

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sar_rag.final_evaluation import (  # noqa: E402
    ANSWER_SCHEMA, AnswerResponse, CheckpointRunner, DeterministicFakeAnswerer,
    build_messages, file_sha256, logical_key, object_sha256, token_f1,
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_frozen_inputs(protocol: dict, manifest: dict, requests: list[dict]) -> None:
    if protocol["status"] != "FROZEN_BEFORE_ANY_PAID_EXECUTION":
        raise ValueError("protocol is not frozen")
    if manifest["status"] != "FROZEN_BEFORE_PAID_SMOKE":
        raise ValueError("smoke manifest is not frozen")
    if manifest["protocol_fingerprint"] != protocol["protocol_fingerprint"]:
        raise ValueError("protocol/smoke fingerprint mismatch")
    expected = {(item["example_id"], system) for item in manifest["examples"] for system in manifest["systems"]}
    actual = {(row["question_id"], row["system"]) for row in requests}
    if len(requests) != manifest["maximum_calls"] or actual != expected:
        raise ValueError("request set differs from frozen smoke manifest")


def dry_metrics(checkpoint: Path, gold_path: Path) -> dict:
    records = {row["logical_key"]: row for row in load_jsonl(checkpoint / "ANSWERS.jsonl")}
    gold = {row["question_id"]: row for row in load_jsonl(gold_path)}
    f1_values, correct_answerability = [], []
    for record in records.values():
        label = gold[record["question_id"]]
        response = record["response"]
        f1_values.append(token_f1(response["rendered_answer"], label["gold_answers"]))
        correct_answerability.append(response["answerability"] == label["gold_answerability"])
    return {
        "records_ingested": len(records),
        "token_f1_field_computed": len(f1_values) == len(records),
        "answerability_field_computed": len(correct_answerability) == len(records),
        "note": "Fake-answer metrics validate ingestion only and are not research results.",
    }


class GroqAdapter:
    def __init__(self, client, attempt_log_path: Path,
                 minimum_attempt_interval_seconds: float = 0.0) -> None:
        self.client = client
        self.attempt_log_path = attempt_log_path
        self.minimum_attempt_interval_seconds = minimum_attempt_interval_seconds
        self.last_attempt_started: float | None = None

    def answer(self, request: dict, generation: dict) -> AnswerResponse:
        messages, _ = build_messages(request["question"], request["evidence"])
        if self.last_attempt_started is not None:
            elapsed = perf_counter() - self.last_attempt_started
            sleep(max(0.0, self.minimum_attempt_interval_seconds - elapsed))
        self.last_attempt_started = perf_counter()
        usage_before, attempts_before = len(self.client.usage_log), len(self.client.attempt_log)
        try:
            raw = self.client.complete(
                messages, "frozen_final_answer", ANSWER_SCHEMA,
                int(generation["decoding"]["max_output_tokens"]),
            )
        finally:
            new_attempts = self.client.attempt_log[attempts_before:]
            with self.attempt_log_path.open("a", encoding="utf-8") as stream:
                for attempt in new_attempts:
                    stream.write(json.dumps({
                        "logical_key": logical_key(request),
                        "request_fingerprint": object_sha256({
                            "dataset": request["dataset"], "question_id": request["question_id"],
                            "system": request["system"], "attempt": attempt,
                        }),
                        **attempt,
                    }, sort_keys=True) + "\n")
        if len(self.client.usage_log) != usage_before + 1:
            raise RuntimeError("successful Groq response lacks exactly one usage record")
        usage = self.client.usage_log[-1]
        attempts = self.client.attempt_log[attempts_before:]
        return AnswerResponse(
            json.loads(raw), int(usage["prompt_tokens"]), int(usage["completion_tokens"]),
            float(usage["latency_seconds"]) * 1000, max(0, len(attempts) - 1),
        )


def paid_authorized(args, protocol: dict, manifest: dict, request_manifest: dict) -> dict:
    if not args.execute_paid:
        raise SystemExit("PAID EXECUTION BLOCKED: --execute-paid was not provided")
    if not args.authorization_file or not args.authorization_file.is_file():
        raise SystemExit("PAID EXECUTION BLOCKED: matching authorization file is absent")
    authorization = json.loads(args.authorization_file.read_text(encoding="utf-8"))
    amendment_path = ROOT / "output/final_evaluation/PAID_SMOKE_AUTHORIZATION_AMENDMENT_001.json"
    expected = {
        "status": "EXPLICIT_USER_AUTHORIZATION_RECORDED",
        "authorized_stage": "paid_smoke_only",
        "user_authorized": True,
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "smoke_manifest_fingerprint": manifest["manifest_fingerprint"],
        "request_manifest_fingerprint": request_manifest["manifest_fingerprint"],
        "authorization_amendment_sha256": file_sha256(amendment_path),
        "maximum_total_api_attempts_including_retries": 48,
        "maximum_retries_per_request": 0,
        "maximum_cost_usd": 0.0432,
    }
    if authorization != expected:
        raise SystemExit("PAID EXECUTION BLOCKED: authorization content/fingerprint mismatch")
    return authorization


def preflight_input_tokens(requests: list[dict]) -> dict:
    from transformers import AutoTokenizer
    tokenizer_path = ROOT / "models/gpt-oss-120b-tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    schema = json.dumps(ANSWER_SCHEMA, separators=(",", ":"))
    counts = []
    for request in requests:
        messages, _ = build_messages(request["question"], request["evidence"])
        count = len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
        count += len(tokenizer.encode(schema, add_special_tokens=False))
        counts.append(count)
    return {"counts": counts, "total": sum(counts), "maximum": max(counts)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("dry-run", "paid-smoke"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--requests", type=Path, default=Path("output/final_evaluation/SMOKE_TEST_REQUESTS.jsonl"))
    parser.add_argument("--gold", type=Path, default=Path("output/final_evaluation/SMOKE_TEST_GOLD_DESIGN_EXPOSED.jsonl"))
    parser.add_argument("--request-manifest", type=Path, default=Path("output/final_evaluation/SMOKE_TEST_REQUESTS_MANIFEST.json"))
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--authorization-file", type=Path)
    parser.add_argument("--execute-paid", action="store_true")
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    request_manifest = json.loads(args.request_manifest.read_text(encoding="utf-8"))
    requests = load_jsonl(args.requests)
    validate_frozen_inputs(protocol, manifest, requests)
    if file_sha256(args.requests) != request_manifest["requests_sha256"]:
        raise ValueError("request file hash differs from frozen request manifest")
    generation = {
        "model": protocol["generation"]["model"],
        "decoding": {
            key: protocol["generation"][key]
            for key in ("temperature", "top_p", "reasoning_effort", "max_output_tokens", "response_format")
        },
    }
    pricing = protocol["pricing"]
    run_fingerprint = object_sha256({
        "mode": args.mode, "protocol": protocol["protocol_fingerprint"],
        "manifest": manifest["manifest_fingerprint"], "requests": file_sha256(args.requests),
    })

    if args.mode == "paid-smoke":
        authorization = paid_authorized(args, protocol, manifest, request_manifest)
        preflight = preflight_input_tokens(requests)
        if preflight["total"] != 68_649 or preflight["maximum"] != 1_988:
            raise SystemExit("PAID EXECUTION BLOCKED: exact tokenizer preflight differs from authorization amendment")
        max_cost = (
            preflight["total"] * pricing["input_per_million"]
            + len(requests) * generation["decoding"]["max_output_tokens"] * pricing["output_per_million"]
        ) / 1_000_000
        if max_cost > authorization["maximum_cost_usd"]:
            raise SystemExit("PAID EXECUTION BLOCKED: preflight cost exceeds authorization ceiling")
        from sar_rag.environment import load_project_env
        load_project_env()
        if not os.environ.get("GROQ_API_KEY"):
            raise SystemExit("PAID EXECUTION BLOCKED: GROQ_API_KEY is unavailable")
        from sar_rag.groq import GroqClient
        client = GroqClient(
            generation["model"], timeout_seconds=protocol["generation"]["timeout_seconds"],
            max_retries=authorization["maximum_retries_per_request"],
        )
        runner = CheckpointRunner(args.checkpoint_dir, run_fingerprint, generation, pricing)
        attempt_log_path = args.checkpoint_dir / "API_ATTEMPTS.jsonl"
        prior_attempts = len(attempt_log_path.read_text(encoding="utf-8").splitlines()) if attempt_log_path.exists() else 0
        remaining_attempts = authorization["maximum_total_api_attempts_including_retries"] - prior_attempts
        if remaining_attempts <= 0:
            raise SystemExit("PAID EXECUTION BLOCKED: cumulative 48-attempt ceiling is exhausted")
        failed_keys: set[str] = set()
        if runner.failures_path.exists():
            failed_keys = {
                row["logical_key"] for row in load_jsonl(runner.failures_path)
                if row["logical_key"] not in runner.records
            }
        pending = [
            request for request in requests
            if logical_key(request) not in runner.records
            and logical_key(request) not in failed_keys
        ]
        scheduled = pending[:remaining_attempts]
        # The account reported an 8,000-token/minute limit during the first run.
        # Start calls at least 20 seconds apart to remain conservatively below it;
        # this is operational pacing only and does not alter frozen prompts/decoding.
        answerer = GroqAdapter(client, attempt_log_path, minimum_attempt_interval_seconds=20.0)
        completed_before = len(runner.records)
        failures_this_resume = 0
        for request in scheduled:
            try:
                runner.run([request], answerer)
            except Exception:
                # The attempt and sanitized failure are already checkpointed.
                # Continue to maximize unique-pair coverage within the global cap.
                failures_this_resume += 1
        summary = {
            "requested": len(scheduled),
            "new_calls": len(runner.records) - completed_before,
            "cache_hits": 0,
            "failed_calls": failures_this_resume,
            "checkpoint_records": len(runner.records),
            "previously_failed_unanswered_skipped": len(failed_keys),
        }
        cumulative_attempts = prior_attempts + len(client.attempt_log)
        if cumulative_attempts > authorization["maximum_total_api_attempts_including_retries"]:
            raise RuntimeError("global API attempt ceiling violated")
        print(json.dumps({
            "status": "PAID_SMOKE_EXECUTED" if summary["checkpoint_records"] == 48 else "PAID_SMOKE_INCOMPLETE",
            **summary, "cumulative_api_attempts": cumulative_attempts,
            "attempt_ceiling": authorization["maximum_total_api_attempts_including_retries"],
            "preflight_input_tokens": preflight["total"], "preflight_cost_ceiling_usd": max_cost,
        }, sort_keys=True))
        return 0

    answerer = DeterministicFakeAnswerer()
    runner = CheckpointRunner(args.checkpoint_dir, run_fingerprint, generation, pricing)
    first = runner.run(requests, answerer)
    second = runner.run(requests, answerer)
    fingerprint_rejection = False
    changed_generation = json.loads(json.dumps(generation))
    changed_generation["decoding"]["max_output_tokens"] += 1
    try:
        CheckpointRunner(args.checkpoint_dir, run_fingerprint, changed_generation, pricing)
    except ValueError:
        fingerprint_rejection = True
    validation = {
        "status": "PASSED_DETERMINISTIC_FAKE_ANSWERER_DRY_RUN",
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "smoke_manifest_fingerprint": manifest["manifest_fingerprint"],
        "request_manifest_fingerprint": request_manifest["manifest_fingerprint"],
        "question_count": manifest["question_count"], "system_count": len(manifest["systems"]),
        "expected_calls": manifest["maximum_calls"], "fake_answerer_calls": answerer.calls,
        "first_run": first, "resume_run": second,
        "resume_zero_duplicate_calls": second["new_calls"] == 0 and second["cache_hits"] == 48,
        "fingerprint_mismatch_rejected": fingerprint_rejection,
        "checkpoint_created": (args.checkpoint_dir / "CHECKPOINT_META.json").is_file() and (args.checkpoint_dir / "ANSWERS.jsonl").is_file(),
        "schema_and_answerability_parsing": True,
        "usage_latency_retry_cost_fields": all(
            all(key in row for key in ("input_tokens", "output_tokens", "total_tokens", "latency_ms", "retries", "estimated_cost_usd"))
            for row in runner.records.values()
        ),
        "metrics_ingestion": dry_metrics(args.checkpoint_dir, args.gold),
        "groq_client_imported_or_invoked": False,
        "paid_calls": 0,
    }
    if not all((validation["resume_zero_duplicate_calls"], validation["fingerprint_mismatch_rejected"], validation["checkpoint_created"], validation["usage_latency_retry_cost_fields"])):
        raise RuntimeError("dry-run validation failed")
    output = ROOT / "output/final_evaluation/DRY_RUN_VALIDATION.json"
    output.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(validation, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
