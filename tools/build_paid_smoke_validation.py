"""Build a sanitized operational validation report for the authorized paid smoke."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sar_rag.final_evaluation import (  # noqa: E402
    ANSWER_SCHEMA, build_messages, logical_key, request_fingerprint, validate_answer,
)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile_nearest_rank(values: list[float], proportion: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, int(proportion * (len(ordered) - 1))))]


def main() -> int:
    output_dir = ROOT / "output/final_evaluation"
    checkpoint = output_dir / "paid_smoke_checkpoint"
    attempts = load_jsonl(checkpoint / "API_ATTEMPTS.jsonl")
    answers = load_jsonl(checkpoint / "ANSWERS.jsonl")
    failures = load_jsonl(checkpoint / "FAILURES.jsonl")
    requests = load_jsonl(output_dir / "SMOKE_TEST_REQUESTS.jsonl")
    request_by_key = {logical_key(row): row for row in requests}
    answer_keys = {row["logical_key"] for row in answers}
    attempted_keys = {row["logical_key"] for row in attempts}
    expected_keys = set(request_by_key)
    unresolved_failed_keys = {row["logical_key"] for row in failures} - answer_keys
    never_attempted_keys = expected_keys - attempted_keys

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / "models/gpt-oss-120b-tokenizer", local_files_only=True
    )
    serialized_schema = json.dumps(ANSWER_SCHEMA, separators=(",", ":"))
    input_tokens_by_key = {}
    for key, request in request_by_key.items():
        messages, _ = build_messages(request["question"], request["evidence"])
        input_tokens_by_key[key] = len(
            tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        ) + len(tokenizer.encode(serialized_schema, add_special_tokens=False))
    attempted_preflight_input_tokens = sum(
        input_tokens_by_key[row["logical_key"]] for row in attempts
    )
    protocol = json.loads((output_dir / "FINAL_EVALUATION_PROTOCOL.json").read_text(encoding="utf-8"))
    pricing = protocol["pricing"]
    attempted_worst_case_cost = (
        attempted_preflight_input_tokens * pricing["input_per_million"]
        + len(attempts) * 800 * pricing["output_per_million"]
    ) / 1_000_000

    failure_categories: Counter[str] = Counter()
    for failure in failures:
        error = failure.get("error", "")
        if "Groq HTTP 429" in error:
            failure_categories["http_429_rate_limit"] += 1
        elif "Groq HTTP 400" in error and "json_validate_failed" in error:
            failure_categories["http_400_json_validate_failed"] += 1
        else:
            failure_categories["local_response_validation"] += 1

    meta = json.loads((checkpoint / "CHECKPOINT_META.json").read_text(encoding="utf-8"))
    schema_valid = True
    fingerprints_valid = True
    for answer in answers:
        request = request_by_key[answer["logical_key"]]
        stored_response = dict(answer["response"])
        stored_rendered = stored_response.pop("rendered_answer")
        regenerated = validate_answer(stored_response, len(request["evidence"]))
        schema_valid &= regenerated["rendered_answer"] == stored_rendered
        fingerprints_valid &= answer["request_fingerprint"] == request_fingerprint(
            request, meta["generation"]
        )

    checkpoint_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            checkpoint / "API_ATTEMPTS.jsonl",
            checkpoint / "ANSWERS.jsonl",
            checkpoint / "FAILURES.jsonl",
            checkpoint / "CHECKPOINT_META.json",
        )
        if path.exists()
    )
    secret_absent = re.search(
        r"(?i)(gsk_[A-Za-z0-9]{10,}|authorization\s*[:=]\s*bearer)", checkpoint_text
    ) is None
    latencies = [float(row["latency_ms"]) for row in answers]

    report = {
        "status": "PAID_SMOKE_INCOMPLETE_ATTEMPT_CEILING_REACHED",
        "scope": {
            "paid_smoke_executed": True,
            "held_out_evaluation_executed": False,
            "held_out_checkpoint_created": (output_dir / "paid_heldout_checkpoint").exists(),
            "research_outcome_scoring_performed": False,
        },
        "frozen_configuration": {
            "protocol_fingerprint": "411bf416e795ac63559ce1a387ab77c8c8494c579a35c65125ab9081dbf0703e",
            "smoke_manifest_fingerprint": "43b3b9c787013e03cf284aba6cc584c4f8f8cf9b023e2621e5881055233725c2",
            "request_manifest_fingerprint": "c6aba531e02404a00f903724e7ca4fa2532420c4c1df81ab5aaf0c82c72eda42",
            "model": meta["generation"]["model"],
            "decoding": meta["generation"]["decoding"],
            "request_fingerprints_valid": bool(fingerprints_valid),
        },
        "authorization_and_cost": {
            "attempt_ceiling": 48,
            "attempts_used_including_failures": len(attempts),
            "automatic_retries": 0,
            "authorized_cost_ceiling_usd": 0.0432,
            "planned_distinct_pair_preflight_worst_case_cost_usd": 0.03333735,
            "actual_attempt_mix_preflight_input_tokens": attempted_preflight_input_tokens,
            "actual_attempt_mix_worst_case_cost_usd": round(attempted_worst_case_cost, 8),
            "successful_response_estimated_cost_usd": round(
                sum(float(row["estimated_cost_usd"]) for row in answers), 8
            ),
            "billing_note": "Failed-request billing is not exposed in the checkpoint; consult Groq billing for invoiced cost.",
        },
        "coverage": {
            "expected_pairs": len(expected_keys),
            "unique_pairs_attempted": len(attempted_keys),
            "validated_responses": len(answers),
            "unanswered_failed_pairs": len(unresolved_failed_keys),
            "previously_failed_pairs_later_completed": len({row["logical_key"] for row in failures} & answer_keys),
            "never_attempted_pairs": len(never_attempted_keys),
            "responses_by_system": dict(sorted(Counter(row["system"] for row in answers).items())),
            "responses_by_dataset": dict(sorted(Counter(row["dataset"] for row in answers).items())),
            "never_attempted_logical_keys": sorted(never_attempted_keys),
        },
        "failures": {
            "failure_records": len(failures),
            "categories": dict(sorted(failure_categories.items())),
            "attempt_statuses": {
                f"{status}|{code}": count
                for (status, code), count in sorted(
                    Counter((row.get("status"), row.get("code")) for row in attempts).items(),
                    key=lambda item: str(item[0]),
                )
            },
        },
        "usage_for_validated_responses": {
            "input_tokens": sum(int(row["input_tokens"]) for row in answers),
            "output_tokens": sum(int(row["output_tokens"]) for row in answers),
            "total_tokens": sum(int(row["total_tokens"]) for row in answers),
            "median_latency_ms": round(statistics.median(latencies), 2),
            "p95_latency_ms": round(percentile_nearest_rank(latencies, 0.95), 2),
            "maximum_latency_ms": round(max(latencies), 2),
        },
        "integrity": {
            "all_saved_responses_schema_valid": schema_valid,
            "checkpoint_secret_pattern_absent": secret_absent,
            "attempt_ceiling_respected": len(attempts) == 48,
            "no_saved_response_reports_retries": all(int(row["retries"]) == 0 for row in answers),
        },
        "interpretation": "Operational smoke evidence only; incomplete and not valid for system-effect conclusions.",
    }
    if not all((
        report["integrity"]["all_saved_responses_schema_valid"],
        report["integrity"]["checkpoint_secret_pattern_absent"],
        report["integrity"]["attempt_ceiling_respected"],
        report["integrity"]["no_saved_response_reports_retries"],
        report["frozen_configuration"]["request_fingerprints_valid"],
        not report["scope"]["held_out_checkpoint_created"],
    )):
        raise RuntimeError("paid smoke operational validation failed")
    destination = output_dir / "PAID_SMOKE_VALIDATION.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
