"""Seal and validate the paid-smoke recovery outcome without scoring answers."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sar_rag.final_evaluation import file_sha256, logical_key, request_fingerprint, validate_answer  # noqa: E402
from sar_rag.smoke_recovery import load_jsonl  # noqa: E402


def main() -> int:
    output = ROOT / "output/final_evaluation"
    recovery = output / "recovery_v1"
    checkpoint = recovery / "checkpoint"
    manifest = json.loads((recovery / "RECOVERY_MANIFEST.json").read_text(encoding="utf-8"))
    meta = json.loads((checkpoint / "CHECKPOINT_META.json").read_text(encoding="utf-8"))
    requests = load_jsonl(output / "SMOKE_TEST_REQUESTS.jsonl")
    original_answers = load_jsonl(output / "paid_smoke_checkpoint/ANSWERS.jsonl")
    attempts = load_jsonl(checkpoint / "API_ATTEMPTS.jsonl")
    answers = load_jsonl(checkpoint / "ANSWERS.jsonl")
    failures = load_jsonl(checkpoint / "FAILURES.jsonl")
    request_by_key = {logical_key(row): row for row in requests}
    eligible_keys = {row["logical_key"] for row in manifest["pairs"]}
    answer_keys = {row["logical_key"] for row in answers}
    unresolved = sorted(eligible_keys - answer_keys)

    all_answers_valid = True
    fingerprints_valid = True
    for answer in answers:
        request = request_by_key[answer["logical_key"]]
        payload = dict(answer["response"])
        rendered = payload.pop("rendered_answer")
        regenerated = validate_answer(payload, len(request["evidence"]))
        all_answers_valid &= regenerated["rendered_answer"] == rendered
        fingerprints_valid &= answer["request_fingerprint"] == request_fingerprint(
            request, meta["generation"]
        )

    original_hashes_match = all(
        file_sha256(output / name) == expected
        for name, expected in manifest["original_integrity_sha256"].items()
    )
    reserved_cost = sum(float(row["authorized_worst_case_cost_usd"]) for row in attempts)
    recorded_input = sum(int(row["input_tokens"]) for row in attempts)
    recorded_output = sum(int(row["output_tokens"]) for row in attempts)
    recorded_cost = (
        recorded_input * manifest["pricing"]["input_per_million"]
        + recorded_output * manifest["pricing"]["output_per_million"]
    ) / 1_000_000
    secret_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in checkpoint.iterdir() if path.is_file()
    )
    secret_absent = re.search(
        r"(?i)(gsk_[A-Za-z0-9]{10,}|authorization\s*[:=]\s*bearer|org_[A-Za-z0-9]{10,})",
        secret_text,
    ) is None
    attempts_by_key = Counter(row["logical_key"] for row in attempts)
    report = {
        "status": "RECOVERY_INCOMPLETE_PER_PAIR_RETRY_LIMIT_REACHED",
        "scope": {
            "eligible_pairs": len(eligible_keys),
            "validated_recovery_pairs": len(answer_keys),
            "unresolved_pairs": unresolved,
            "combined_original_plus_recovery_validated_pairs": len(
                {row["logical_key"] for row in original_answers} | answer_keys
            ),
            "held_out_evaluation_executed": False,
            "research_performance_scoring_performed": False,
        },
        "attempts": {
            "total": len(attempts),
            "maximum_authorized": manifest["maximum_total_api_attempts_including_retries"],
            "remaining_global_allowance": (
                manifest["maximum_total_api_attempts_including_retries"] - len(attempts)
            ),
            "by_pair": dict(sorted(attempts_by_key.items())),
            "transport_statuses": dict(sorted(Counter(row["status"] for row in attempts).items())),
            "failure_statuses": dict(sorted(Counter(row["status"] for row in failures).items())),
            "unresolved_pairs_exhausted_three_attempt_limit": all(
                attempts_by_key[key] == manifest["maximum_attempts_per_pair_including_initial"]
                for key in unresolved
            ),
        },
        "usage_and_cost": {
            "recorded_input_tokens_all_http_successes": recorded_input,
            "recorded_output_tokens_all_http_successes": recorded_output,
            "recorded_token_cost_usd": round(recorded_cost, 8),
            "validated_answer_input_tokens": sum(int(row["input_tokens"]) for row in answers),
            "validated_answer_output_tokens": sum(int(row["output_tokens"]) for row in answers),
            "validated_answer_estimated_cost_usd": round(
                sum(float(row["estimated_cost_usd"]) for row in answers), 8
            ),
            "cumulative_authorized_worst_case_cost_usd": round(reserved_cost, 8),
            "authorized_cost_ceiling_usd": manifest["maximum_cost_usd"],
            "remaining_conservative_cost_allowance_usd": round(
                manifest["maximum_cost_usd"] - reserved_cost, 8
            ),
            "billing_note": "Seven HTTP 400 attempts expose no usage; invoiced cost must be checked in Groq billing.",
        },
        "validated_pairs": [
            {
                "logical_key": row["logical_key"],
                "attempt_for_pair": row["attempt_for_pair"],
                "input_tokens": row["input_tokens"],
                "output_tokens": row["output_tokens"],
                "estimated_cost_usd": row["estimated_cost_usd"],
            }
            for row in answers
        ],
        "integrity_and_operational_checks": {
            "original_artifact_hashes_match": original_hashes_match,
            "generation_configuration_unchanged": meta["generation"] == json.loads(
                (output / "paid_smoke_checkpoint/CHECKPOINT_META.json").read_text(encoding="utf-8")
            )["generation"],
            "recovery_scope_exact": set(attempts_by_key) <= eligible_keys,
            "attempt_ceiling_respected": len(attempts) <= manifest[
                "maximum_total_api_attempts_including_retries"
            ],
            "cost_ceiling_respected_conservatively": reserved_cost <= manifest["maximum_cost_usd"] + 1e-12,
            "saved_recovery_responses_valid": all_answers_valid,
            "saved_request_fingerprints_valid": bool(fingerprints_valid),
            "secret_pattern_absent": secret_absent,
            "all_48_pairs_validated": len({row["logical_key"] for row in original_answers} | answer_keys) == 48,
            "operational_smoke_checks_passed": False,
        },
        "interpretation": "Operational recovery only; incomplete and not evidence of research superiority.",
    }
    required = report["integrity_and_operational_checks"]
    if not all((
        required["original_artifact_hashes_match"],
        required["generation_configuration_unchanged"],
        required["recovery_scope_exact"],
        required["attempt_ceiling_respected"],
        required["cost_ceiling_respected_conservatively"],
        required["saved_recovery_responses_valid"],
        required["saved_request_fingerprints_valid"],
        required["secret_pattern_absent"],
        not required["all_48_pairs_validated"],
    )):
        raise RuntimeError("recovery outcome integrity validation failed")
    destination = recovery / "RECOVERY_VALIDATION.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
