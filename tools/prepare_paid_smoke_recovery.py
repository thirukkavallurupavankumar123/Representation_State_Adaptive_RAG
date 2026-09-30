"""Freeze the protocol-preserving paid-smoke recovery manifest and cost bound."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sar_rag.final_evaluation import (  # noqa: E402
    ANSWER_SCHEMA, build_messages, file_sha256, logical_key, object_sha256,
)
from sar_rag.smoke_recovery import missing_requests  # noqa: E402


ORIGINAL_HASHES = {
    "FINAL_EVALUATION_PROTOCOL.json": "d96aaaa9066258a6a5291409d56858417b8415b18e4369739ca177115308cf82",
    "SMOKE_TEST_MANIFEST.json": "d43d0f65befd192f89d0647548760f76066473d5e93e0d8b3dbababfa3f73675",
    "SMOKE_TEST_REQUESTS.jsonl": "509d74568a6f34584a0931eb57db4b63dd4edaf1f2b3053e5f3387f36b47e462",
    "SMOKE_TEST_REQUESTS_MANIFEST.json": "052945e1db3c0e30332cde4b917a0f39bc55231ee67d115730690f9cf5e1b617",
    "paid_smoke_checkpoint/CHECKPOINT_META.json": "fb471cf1af02120ca4bf862994e806b82597dd7a17c544d75ddc7320744c63d4",
    "paid_smoke_checkpoint/API_ATTEMPTS.jsonl": "c68ff3f07bcc57365855894747c68967d7e13fd9659127080006123c28d5b6c5",
    "paid_smoke_checkpoint/ANSWERS.jsonl": "e44ad14bf932976f26b7f03c1f76610169b55480f74bdb1d2b21a9ee7de275dd",
    "paid_smoke_checkpoint/FAILURES.jsonl": "44eae81a31adc899f7507dbf4d84305d6f5797745863172a3254823edc936ea3",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def failure_category(errors: list[str]) -> str:
    if any("HTTP 400" in error and "json_validate_failed" in error for error in errors):
        return "http_400_json_validate_failed"
    if any("claim is empty, uncited, or cites unknown evidence" in error for error in errors):
        return "local_invalid_or_missing_citation"
    if any("insufficient output must not contain factual claims" in error for error in errors):
        return "local_insufficient_with_claims"
    return "not_attempted_attempt_ceiling"


def main() -> int:
    output = ROOT / "output/final_evaluation"
    recovery = output / "recovery_v1"
    recovery.mkdir(parents=True, exist_ok=True)
    observed_hashes = {name: file_sha256(output / name) for name in ORIGINAL_HASHES}
    if observed_hashes != ORIGINAL_HASHES:
        raise RuntimeError("original paid-smoke evidence changed; refusing recovery preparation")

    requests = load_jsonl(output / "SMOKE_TEST_REQUESTS.jsonl")
    answers = load_jsonl(output / "paid_smoke_checkpoint/ANSWERS.jsonl")
    attempts = load_jsonl(output / "paid_smoke_checkpoint/API_ATTEMPTS.jsonl")
    failures = load_jsonl(output / "paid_smoke_checkpoint/FAILURES.jsonl")
    missing = missing_requests(requests, answers)
    if len(missing) != 8:
        raise RuntimeError(f"expected eight missing pairs, found {len(missing)}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / "models/gpt-oss-120b-tokenizer", local_files_only=True
    )
    schema_text = json.dumps(ANSWER_SCHEMA, separators=(",", ":"))
    attempts_by_key = Counter(row["logical_key"] for row in attempts)
    errors_by_key: dict[str, list[str]] = {}
    for row in failures:
        errors_by_key.setdefault(row["logical_key"], []).append(row["error"])

    pairs = []
    total_input_tokens_once = 0
    for request in missing:
        key = logical_key(request)
        messages, _ = build_messages(request["question"], request["evidence"])
        input_tokens = len(
            tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        ) + len(tokenizer.encode(schema_text, add_special_tokens=False))
        total_input_tokens_once += input_tokens
        pairs.append({
            "logical_key": key,
            "dataset": request["dataset"],
            "question_id": request["question_id"],
            "system": request["system"],
            "question": request["question"],
            "evidence_count": len(request["evidence"]),
            "preflight_input_tokens": input_tokens,
            "original_attempts": attempts_by_key[key],
            "original_failure_category": failure_category(errors_by_key.get(key, [])),
        })

    protocol = json.loads((output / "FINAL_EVALUATION_PROTOCOL.json").read_text(encoding="utf-8"))
    maximum_attempts_per_pair = 3
    maximum_total_attempts = len(pairs) * maximum_attempts_per_pair
    maximum_output_tokens = protocol["generation"]["max_output_tokens"]
    pricing = protocol["pricing"]
    maximum_cost = (
        total_input_tokens_once * maximum_attempts_per_pair * pricing["input_per_million"]
        + maximum_total_attempts * maximum_output_tokens * pricing["output_per_million"]
    ) / 1_000_000
    manifest_body = {
        "version": "paid-smoke-recovery-v1",
        "status": "FROZEN_AWAITING_SEPARATE_AUTHORIZATION",
        "scope": "only the eight system-question pairs missing from the original paid smoke",
        "original_integrity_sha256": observed_hashes,
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "configuration_change": False,
        "prompt_schema_model_decoding_unchanged": True,
        "consistent_full_48_pair_rerun_required": False,
        "maximum_attempts_per_pair_including_initial": maximum_attempts_per_pair,
        "maximum_retries_per_pair": maximum_attempts_per_pair - 1,
        "maximum_total_api_attempts_including_retries": maximum_total_attempts,
        "minimum_seconds_between_attempt_starts": 30,
        "retry_order": "round-robin: all missing pairs once before second or third attempts",
        "maximum_output_tokens": maximum_output_tokens,
        "total_preflight_input_tokens_one_attempt_each": total_input_tokens_once,
        "maximum_cost_usd": round(maximum_cost, 8),
        "pricing": pricing,
        "pairs": pairs,
        "prohibitions": [
            "do not modify or append to the original paid_smoke_checkpoint",
            "do not run held-out evaluation",
            "do not score or draw research-performance conclusions",
        ],
    }
    manifest_body["manifest_fingerprint"] = object_sha256(manifest_body)
    manifest_path = recovery / "RECOVERY_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest_body, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    integrity = {
        "status": "ORIGINAL_PAID_SMOKE_FILES_VERIFIED_IMMUTABLE",
        "verified_sha256": observed_hashes,
        "recovery_manifest_sha256": file_sha256(manifest_path),
    }
    (recovery / "ORIGINAL_INTEGRITY.json").write_text(
        json.dumps(integrity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    authorization_template = {
        "status": "AWAITING_EXPLICIT_USER_AUTHORIZATION",
        "authorized_stage": "paid_smoke_recovery_v1_only",
        "user_authorized": False,
        "recovery_manifest_fingerprint": manifest_body["manifest_fingerprint"],
        "maximum_total_api_attempts_including_retries": maximum_total_attempts,
        "maximum_retries_per_pair": maximum_attempts_per_pair - 1,
        "maximum_cost_usd": round(maximum_cost, 8),
    }
    (recovery / "AUTHORIZATION_TEMPLATE.json").write_text(
        json.dumps(authorization_template, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "missing_pairs": len(pairs),
        "maximum_total_attempts": maximum_total_attempts,
        "maximum_cost_usd": round(maximum_cost, 8),
        "manifest_fingerprint": manifest_body["manifest_fingerprint"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
