"""Seal the completed v1.1 operational smoke without scoring answer quality."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sar_rag.final_evaluation import file_sha256, logical_key, object_sha256, validate_answer  # noqa: E402
from sar_rag.generation_v1_1 import request_fingerprint_v1_1  # noqa: E402
from sar_rag.smoke_recovery import load_jsonl  # noqa: E402
from sar_rag.smoke_v1_1 import V11SmokeRunner  # noqa: E402


class NeverCall:
    def attempt(self, request: dict):  # pragma: no cover - a call is a failure
        raise AssertionError("resume attempted a duplicate API call")


def main() -> int:
    output = ROOT / "output/final_evaluation"
    base = output / "generation_v1_1"
    checkpoint = base / "checkpoint"
    protocol_path = base / "PROTOCOL.json"
    manifest_path = base / "RUN_MANIFEST.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    requests = load_jsonl(output / "SMOKE_TEST_REQUESTS.jsonl")
    request_by_key = {logical_key(row): row for row in requests}
    attempts = load_jsonl(checkpoint / "API_ATTEMPTS.jsonl")
    answers = load_jsonl(checkpoint / "ANSWERS.jsonl")
    failures = load_jsonl(checkpoint / "FAILURES.jsonl")
    answer_by_key = {row["logical_key"]: row for row in answers}

    schema_valid = True
    fingerprints_valid = True
    for answer in answers:
        request = request_by_key[answer["logical_key"]]
        payload = dict(answer["response"])
        rendered = payload.pop("rendered_answer")
        regenerated = validate_answer(payload, len(request["evidence"]))
        schema_valid &= regenerated["rendered_answer"] == rendered
        fingerprints_valid &= answer["request_fingerprint"] == request_fingerprint_v1_1(request)

    protocol_fingerprint_valid = object_sha256({
        key: value for key, value in protocol.items() if key != "protocol_fingerprint"
    }) == protocol["protocol_fingerprint"]
    manifest_fingerprint_valid = object_sha256({
        key: value for key, value in manifest.items() if key != "manifest_fingerprint"
    }) == manifest["manifest_fingerprint"]
    preserved_hashes_match = all(
        file_sha256(output / name) == expected
        for name, expected in protocol["preserved_artifact_sha256"].items()
    )
    resume = V11SmokeRunner(
        checkpoint, protocol, manifest,
        protocol_path=protocol_path, manifest_path=manifest_path,
    ).run(requests, NeverCall())

    pricing = protocol["cost"]["pricing"]
    input_tokens = sum(int(row["input_tokens"]) for row in attempts)
    output_tokens = sum(int(row["output_tokens"]) for row in attempts)
    estimated_cost = (
        input_tokens * pricing["input_per_million"]
        + output_tokens * pricing["output_per_million"]
    ) / 1_000_000
    reserved_cost = sum(float(row["authorized_worst_case_cost_usd"]) for row in attempts)
    latencies = [float(row["latency_ms"]) for row in attempts]
    checkpoint_text = "\n".join(
        path.read_text(encoding="utf-8") for path in checkpoint.iterdir() if path.is_file()
    )
    secret_absent = re.search(
        r"(?i)(gsk_[A-Za-z0-9]{10,}|authorization\s*[:=]\s*bearer|org_[A-Za-z0-9]{10,})",
        checkpoint_text,
    ) is None

    report = {
        "status": "PASSED_OPERATIONAL_SMOKE_V1_1",
        "scope": {
            "pairs": 48,
            "systems": 3,
            "questions": 16,
            "responses_by_system": dict(sorted(Counter(row["system"] for row in answers).items())),
            "responses_by_dataset": dict(sorted(Counter(row["dataset"] for row in answers).items())),
            "held_out_evaluation_executed": False,
            "research_performance_scoring_performed": False,
        },
        "validity": {
            "first_attempt_valid_pairs": sum(bool(row["first_attempt_valid"]) for row in answers),
            "eventually_valid_pairs": sum(bool(row["eventually_valid"]) for row in answers),
            "unresolved_pairs": sorted(set(request_by_key) - set(answer_by_key)),
            "failure_records": len(failures),
        },
        "attempts": {
            "total": len(attempts),
            "maximum_authorized": manifest["maximum_total_api_attempts_including_retries"],
            "maximum_per_pair": manifest["maximum_attempts_per_pair_including_first"],
            "attempts_per_pair_distribution": dict(sorted(Counter(
                Counter(row["logical_key"] for row in attempts).values()
            ).items())),
            "transport_statuses": dict(sorted(Counter(row["transport_status"] for row in attempts).items())),
            "remaining_authorized_attempts": manifest["maximum_total_api_attempts_including_retries"] - len(attempts),
        },
        "usage_and_cost": {
            "recorded_input_tokens": input_tokens,
            "recorded_output_tokens": output_tokens,
            "recorded_total_tokens": input_tokens + output_tokens,
            "estimated_charge_usd_from_recorded_usage_and_published_rates": round(estimated_cost, 8),
            "conservative_reserved_cost_usd": round(reserved_cost, 8),
            "authorized_cost_ceiling_usd": manifest["maximum_cost_usd"],
            "remaining_conservative_allowance_usd": round(manifest["maximum_cost_usd"] - reserved_cost, 8),
            "actual_invoiced_charge_usd": None,
            "actual_invoice_note": "Not available from API logs; verify in Groq billing. Estimated charge is not an invoice.",
        },
        "latency_ms": {
            "median": round(statistics.median(latencies), 2),
            "maximum": round(max(latencies), 2),
        },
        "integrity_and_operational_checks": {
            "protocol_fingerprint_valid": protocol_fingerprint_valid,
            "manifest_fingerprint_valid": manifest_fingerprint_valid,
            "protocol_manifest_link_valid": manifest["protocol_fingerprint"] == protocol["protocol_fingerprint"],
            "preserved_artifact_hashes_match": preserved_hashes_match,
            "all_saved_responses_schema_and_semantic_valid": schema_valid,
            "all_saved_request_fingerprints_valid": bool(fingerprints_valid),
            "all_pairs_valid": len(answer_by_key) == 48,
            "all_pairs_first_attempt_valid": all(row["attempt_for_pair"] == 1 for row in answers),
            "attempt_ceiling_respected": len(attempts) <= manifest["maximum_total_api_attempts_including_retries"],
            "per_pair_ceiling_respected": all(
                count <= manifest["maximum_attempts_per_pair_including_first"]
                for count in Counter(row["logical_key"] for row in attempts).values()
            ),
            "conservative_cost_ceiling_respected": reserved_cost <= manifest["maximum_cost_usd"] + 1e-12,
            "resume_created_zero_duplicate_calls": resume["new_attempts"] == 0 and resume["new_answers"] == 0,
            "secret_pattern_absent": secret_absent,
            "heldout_checkpoint_absent": not (output / "paid_heldout_checkpoint").exists(),
            "operational_smoke_checks_passed": True,
        },
        "artifacts": {
            "attempt_ledger_sha256": file_sha256(checkpoint / "API_ATTEMPTS.jsonl"),
            "answers_sha256": file_sha256(checkpoint / "ANSWERS.jsonl"),
            "checkpoint_meta_sha256": file_sha256(checkpoint / "CHECKPOINT_META.json"),
        },
        "interpretation": "Operational smoke passed. No answer-quality or system-superiority conclusion is drawn.",
    }
    checks = report["integrity_and_operational_checks"]
    if not all(checks.values()) or report["validity"]["unresolved_pairs"]:
        raise RuntimeError("v1.1 operational validation failed")
    destination = base / "VALIDATION.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
