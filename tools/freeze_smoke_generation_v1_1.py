"""Freeze smoke-generation v1.1 and its fresh 48-pair execution manifest."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sar_rag.final_evaluation import ANSWER_SCHEMA, file_sha256, logical_key, object_sha256  # noqa: E402
from sar_rag.generation_v1_1 import (  # noqa: E402
    MAX_COMPLETION_TOKENS, PROTOCOL_VERSION, build_chat_payload_v1_1,
    generation_config_v1_1, request_fingerprint_v1_1,
)
from sar_rag.smoke_recovery import load_jsonl  # noqa: E402


PRESERVED_FILES = (
    "FINAL_EVALUATION_PROTOCOL.json",
    "SMOKE_TEST_MANIFEST.json",
    "SMOKE_TEST_REQUESTS.jsonl",
    "SMOKE_TEST_REQUESTS_MANIFEST.json",
    "paid_smoke_checkpoint/CHECKPOINT_META.json",
    "paid_smoke_checkpoint/API_ATTEMPTS.jsonl",
    "paid_smoke_checkpoint/ANSWERS.jsonl",
    "paid_smoke_checkpoint/FAILURES.jsonl",
    "recovery_v1/RECOVERY_MANIFEST.json",
    "recovery_v1/checkpoint/CHECKPOINT_META.json",
    "recovery_v1/checkpoint/API_ATTEMPTS.jsonl",
    "recovery_v1/checkpoint/ANSWERS.jsonl",
    "recovery_v1/checkpoint/FAILURES.jsonl",
    "recovery_v1/RECOVERY_VALIDATION.json",
)


def main() -> int:
    output = ROOT / "output/final_evaluation"
    destination = output / "generation_v1_1"
    destination.mkdir(parents=True, exist_ok=True)
    requests_path = output / "SMOKE_TEST_REQUESTS.jsonl"
    requests = load_jsonl(requests_path)
    if len(requests) != 48 or len({logical_key(row) for row in requests}) != 48:
        raise RuntimeError("v1.1 requires the exact fresh 48-pair smoke set")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / "models/gpt-oss-120b-tokenizer", local_files_only=True
    )
    serialized_schema = json.dumps(ANSWER_SCHEMA, separators=(",", ":"))
    pair_rows = []
    conservative_provider_overhead_tokens_per_attempt = 128
    for request in requests:
        payload = build_chat_payload_v1_1(request["question"], request["evidence"])
        input_tokens = len(tokenizer.apply_chat_template(
            payload["messages"], tokenize=True, add_generation_prompt=True
        )) + len(tokenizer.encode(serialized_schema, add_special_tokens=False))
        pair_rows.append({
            "logical_key": logical_key(request),
            "dataset": request["dataset"],
            "question_id": request["question_id"],
            "system": request["system"],
            "preflight_input_tokens": input_tokens,
            "conservative_billed_input_tokens": (
                input_tokens + conservative_provider_overhead_tokens_per_attempt
            ),
            "request_fingerprint_v1_1": request_fingerprint_v1_1(request),
        })

    pricing = {
        "currency": "USD",
        "input_per_million": 0.15,
        "cached_input_per_million": 0.075,
        "output_per_million": 0.60,
        "source": "https://console.groq.com/docs/model/openai/gpt-oss-120b",
        "verified_date": "2026-09-26",
    }
    attempts_per_pair = 3
    maximum_attempts = len(pair_rows) * attempts_per_pair
    input_tokens_one_round = sum(row["preflight_input_tokens"] for row in pair_rows)
    conservative_input_tokens_one_round = sum(
        row["conservative_billed_input_tokens"] for row in pair_rows
    )
    maximum_cost = (
        conservative_input_tokens_one_round * attempts_per_pair * pricing["input_per_million"]
        + maximum_attempts * MAX_COMPLETION_TOKENS * pricing["output_per_million"]
    ) / 1_000_000
    preserved_hashes = {name: file_sha256(output / name) for name in PRESERVED_FILES}

    protocol_body = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "FROZEN_AWAITING_EXPLICIT_PAID_AUTHORIZATION",
        "purpose": "fresh operational smoke rerun only; not research evidence",
        "source_request_set": str(requests_path.relative_to(ROOT)).replace("\\", "/"),
        "source_request_sha256": file_sha256(requests_path),
        "pair_count": 48,
        "generation": generation_config_v1_1(),
        "schema": ANSWER_SCHEMA,
        "changes_from_v1": [
            "prompt explicitly requires all three top-level fields and states empty-field behavior",
            "max_completion_tokens increased from 800 to 1600",
        ],
        "unchanged_from_v1": [
            "model", "temperature", "top_p", "reasoning_effort", "strict mode",
            "JSON schema", "semantic response validator", "questions", "evidence",
        ],
        "rationale": {
            "confirmed": [
                "one unresolved saved generation omitted required missing_information",
                "one separate saved recovery failure explicitly reported max completion tokens reached",
                "saved valid responses reached 796 of the former 800-token limit",
            ],
            "not_confirmed": [
                "token exhaustion was not established for either unresolved pair",
                "the provider did not return failed_generation for five of six unresolved-pair 400s",
            ],
        },
        "official_documentation_review": {
            "strict_model_supported": True,
            "all_object_fields_required": True,
            "all_objects_closed_with_additionalProperties_false": True,
            "reasoning_effort_supported": True,
            "max_completion_tokens_parameter_current": True,
            "reasoning_format_omitted_because_gpt_oss_does_not_support_it": True,
            "sources": [
                "https://console.groq.com/docs/structured-outputs",
                "https://console.groq.com/docs/reasoning",
                "https://console.groq.com/docs/api-reference",
            ],
            "finding": "request satisfies documented mandatory strict-schema requirements; observed strict-mode 400s remain a provider/model operational issue",
        },
        "retry_policy": {
            "maximum_attempts_per_pair_including_first": attempts_per_pair,
            "maximum_retries_per_pair": attempts_per_pair - 1,
            "maximum_total_attempts_including_retries": maximum_attempts,
            "round_robin": "complete attempt round N for every unresolved pair before round N+1",
            "retryable": [
                "HTTP 429", "HTTP 5xx", "timeout/network error",
                "json_validate_failed/JSON parse failure", "local semantic validation failure",
            ],
            "abort_entire_run_without_retry": [
                "HTTP 401/403 authentication or authorization error",
                "request/configuration rejection not classified as generated-output validation",
                "configuration, manifest, fingerprint, attempt, or cost guard mismatch",
            ],
            "persistent_failure": "retain every attempt; leave pair missing after three attempts; never impute, repair, or reuse another system answer",
            "validity_reporting": [
                "first_attempt_validity per pair",
                "eventual_validity within three attempts per pair",
                "attempt number of first valid response",
            ],
        },
        "rate_limit_policy": {
            "minimum_seconds_between_attempt_starts": 30,
            "honor_retry_after": True,
            "retry_after_safety_margin_seconds": 0.25,
            "every_429_counts_as_an_attempt": True,
        },
        "cost": {
            "pricing": pricing,
            "preflight_input_tokens_one_complete_round": input_tokens_one_round,
            "observed_recovery_prompt_token_delta": 67,
            "conservative_provider_overhead_tokens_per_attempt": conservative_provider_overhead_tokens_per_attempt,
            "conservative_billed_input_tokens_one_complete_round": conservative_input_tokens_one_round,
            "maximum_output_tokens_per_attempt": MAX_COMPLETION_TOKENS,
            "maximum_attempts": maximum_attempts,
            "conservative_maximum_cost_usd": round(maximum_cost, 8),
            "guard": "check remaining attempt and conservative remaining cost allowance before every call",
        },
        "preserved_artifact_sha256": preserved_hashes,
        "prohibitions": [
            "no paid call without a separately matching v1.1 authorization file",
            "no held-out evaluation",
            "no smoke-based research superiority claim",
            "no modification of original or recovery artifacts",
        ],
    }
    protocol_body["protocol_fingerprint"] = object_sha256(protocol_body)
    protocol_path = destination / "PROTOCOL.json"
    protocol_path.write_text(json.dumps(protocol_body, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    run_manifest = {
        "status": "FROZEN_AWAITING_EXPLICIT_PAID_AUTHORIZATION",
        "protocol_fingerprint": protocol_body["protocol_fingerprint"],
        "protocol_sha256": file_sha256(protocol_path),
        "pair_count": 48,
        "maximum_total_api_attempts_including_retries": maximum_attempts,
        "maximum_attempts_per_pair_including_first": attempts_per_pair,
        "maximum_retries_per_pair": attempts_per_pair - 1,
        "maximum_cost_usd": round(maximum_cost, 8),
        "pairs": pair_rows,
    }
    run_manifest["manifest_fingerprint"] = object_sha256(run_manifest)
    manifest_path = destination / "RUN_MANIFEST.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    authorization_template = {
        "status": "AWAITING_EXPLICIT_USER_AUTHORIZATION",
        "authorized_stage": "paid_smoke_generation_v1_1_only",
        "user_authorized": False,
        "protocol_fingerprint": protocol_body["protocol_fingerprint"],
        "run_manifest_fingerprint": run_manifest["manifest_fingerprint"],
        "maximum_total_api_attempts_including_retries": maximum_attempts,
        "maximum_attempts_per_pair_including_first": attempts_per_pair,
        "maximum_cost_usd": round(maximum_cost, 8),
    }
    (destination / "AUTHORIZATION_TEMPLATE.json").write_text(
        json.dumps(authorization_template, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "protocol_fingerprint": protocol_body["protocol_fingerprint"],
        "run_manifest_fingerprint": run_manifest["manifest_fingerprint"],
        "pair_count": 48,
        "maximum_attempts": maximum_attempts,
        "maximum_cost_usd": round(maximum_cost, 8),
        "input_tokens_one_round": input_tokens_one_round,
        "conservative_billed_input_tokens_one_round": conservative_input_tokens_one_round,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
