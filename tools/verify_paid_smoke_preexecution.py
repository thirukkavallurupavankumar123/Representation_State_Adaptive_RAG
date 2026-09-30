"""Fail-closed verification immediately before the authorized Groq smoke test."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sar_rag.environment import load_project_env  # noqa: E402
from sar_rag.final_evaluation import ANSWER_SCHEMA, build_messages, file_sha256  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

OUT = ROOT / "output/final_evaluation"


def load(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def main() -> None:
    protocol = load("FINAL_EVALUATION_PROTOCOL.json")
    smoke = load("SMOKE_TEST_MANIFEST.json")
    request_manifest = load("SMOKE_TEST_REQUESTS_MANIFEST.json")
    authorization = load("PAID_SMOKE_AUTHORIZATION.json")
    amendment_path = OUT / "PAID_SMOKE_AUTHORIZATION_AMENDMENT_001.json"
    requests_path = OUT / "SMOKE_TEST_REQUESTS.jsonl"
    requests = [json.loads(line) for line in requests_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    tokenizer_path = ROOT / "models/gpt-oss-120b-tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    schema = json.dumps(ANSWER_SCHEMA, separators=(",", ":"))
    counts = []
    for request in requests:
        messages, _ = build_messages(request["question"], request["evidence"])
        counts.append(
            len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
            + len(tokenizer.encode(schema, add_special_tokens=False))
        )
    rates = protocol["pricing"]
    cost_ceiling = (
        sum(counts) * rates["input_per_million"]
        + len(requests) * protocol["generation"]["max_output_tokens"] * rates["output_per_million"]
    ) / 1_000_000
    attempt_log = OUT / "paid_smoke_checkpoint/API_ATTEMPTS.jsonl"
    existing_attempts = len(attempt_log.read_text(encoding="utf-8").splitlines()) if attempt_log.exists() else 0
    load_project_env()
    checks = {
        "protocol_frozen": protocol["status"] == "FROZEN_BEFORE_ANY_PAID_EXECUTION",
        "protocol_fingerprint_matches": authorization["protocol_fingerprint"] == protocol["protocol_fingerprint"],
        "smoke_fingerprint_matches": authorization["smoke_manifest_fingerprint"] == smoke["manifest_fingerprint"],
        "request_manifest_fingerprint_matches": authorization["request_manifest_fingerprint"] == request_manifest["manifest_fingerprint"],
        "authorization_amendment_hash_matches": authorization["authorization_amendment_sha256"] == file_sha256(amendment_path),
        "request_hash_matches": request_manifest["requests_sha256"] == file_sha256(requests_path),
        "model_matches": protocol["generation"]["model"] == "openai/gpt-oss-120b",
        "generation_matches": all((
            protocol["generation"]["temperature"] == 0,
            protocol["generation"]["top_p"] == 1,
            protocol["generation"]["reasoning_effort"] == "medium",
            protocol["generation"]["max_output_tokens"] == 800,
        )),
        "exactly_48_requests": len(requests) == 48,
        "zero_retries": authorization["maximum_retries_per_request"] == 0,
        "attempt_ceiling_48": authorization["maximum_total_api_attempts_including_retries"] == 48,
        "no_prior_api_attempts": existing_attempts == 0,
        "preflight_input_total_matches": sum(counts) == 68_649,
        "preflight_input_max_matches": max(counts) == 1_988,
        "cost_within_authorization": cost_ceiling <= authorization["maximum_cost_usd"],
        "api_key_available_not_printed": bool(os.environ.get("GROQ_API_KEY")),
        "heldout_not_requested": True,
    }
    if not all(checks.values()):
        raise RuntimeError("paid smoke pre-execution verification failed: " + ", ".join(key for key, value in checks.items() if not value))
    payload = {
        "status": "VERIFIED_READY_FOR_AUTHORIZED_48_ATTEMPT_SMOKE",
        "checks": checks,
        "model": protocol["generation"]["model"],
        "request_count": len(requests),
        "existing_api_attempts": existing_attempts,
        "maximum_total_api_attempts_including_retries": 48,
        "maximum_retries_per_request": 0,
        "preflight_input_tokens_total": sum(counts),
        "preflight_input_tokens_max_single_request": max(counts),
        "maximum_output_tokens_total": len(requests) * 800,
        "preflight_maximum_cost_usd": cost_ceiling,
        "authorized_cost_ceiling_usd": authorization["maximum_cost_usd"],
        "tokenizer_json_sha256": hashlib.sha256((tokenizer_path / "tokenizer.json").read_bytes()).hexdigest(),
        "api_key_value_recorded": False,
        "heldout_evaluation": "PROHIBITED_NOT_RUN",
    }
    target = OUT / "PAID_SMOKE_PREEXECUTION_VALIDATION.json"
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("existing pre-execution validation differs")
    else:
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "status", "model", "request_count", "existing_api_attempts",
        "maximum_total_api_attempts_including_retries", "preflight_input_tokens_total",
        "preflight_maximum_cost_usd", "authorized_cost_ceiling_usd", "heldout_evaluation",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
