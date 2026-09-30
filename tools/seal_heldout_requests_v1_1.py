"""Validate and seal completed held-out requests without opening gold data."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sar_rag.final_evaluation import file_sha256, logical_key, object_sha256  # noqa: E402
from sar_rag.heldout_v1_1 import heldout_request_fingerprint_v1_1  # noqa: E402
from sar_rag.smoke_recovery import load_jsonl  # noqa: E402


OUT = ROOT / "output/final_evaluation/heldout_v1_1"
PROTOCOL = OUT / "FINAL_EVALUATION_PROTOCOL_AMENDMENT_V1_1.json"
MANIFEST = OUT / "FROZEN_QUESTION_SYSTEM_MANIFEST.json"
REQUESTS = OUT / "REQUESTS.jsonl"
AUDIT = OUT / "RETRIEVAL_AND_PACKING_AUDIT.jsonl"
RAW_REQUEST_MANIFEST = OUT / "REQUEST_MANIFEST.json"
SEALED_REQUEST_MANIFEST = OUT / "REQUEST_MANIFEST_SEALED.json"
REPORT = OUT / "SEALED_REQUEST_VALIDATION.json"
COST_PLAN = OUT / "COST_AND_RUNTIME_PLAN_SEALED_REQUESTS.md"
COMMANDS = OUT / "EXECUTION_AND_SCORING_COMMANDS_SEALED_REQUESTS.md"
AUTHORIZATION = OUT / "PROPOSED_AUTHORIZATION_SEALED_REQUESTS_DISABLED.json"
LIMITS = OUT / "ACCOUNT_RATE_LIMIT_REQUIREMENTS.md"


def assert_new(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite sealed artifacts: {existing}")


def body(value: dict, field: str) -> dict:
    return {key: item for key, item in value.items() if key != field}


def main() -> int:
    outputs = [SEALED_REQUEST_MANIFEST, REPORT, COST_PLAN, COMMANDS, AUTHORIZATION, LIMITS]
    assert_new(outputs)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    raw_manifest = json.loads(RAW_REQUEST_MANIFEST.read_text(encoding="utf-8"))
    requests = load_jsonl(REQUESTS)
    audits = load_jsonl(AUDIT)

    protocol_valid = object_sha256(body(protocol, "protocol_fingerprint")) == protocol["protocol_fingerprint"]
    manifest_valid = object_sha256(body(manifest, "manifest_fingerprint")) == manifest["manifest_fingerprint"]
    raw_manifest_valid = object_sha256(body(raw_manifest, "request_manifest_fingerprint")) == raw_manifest["request_manifest_fingerprint"]
    if not all((protocol_valid, manifest_valid, raw_manifest_valid)):
        raise RuntimeError("one or more frozen fingerprints are invalid")
    if manifest["protocol_fingerprint"] != protocol["protocol_fingerprint"]:
        raise RuntimeError("protocol/manifest linkage mismatch")
    if raw_manifest["protocol_fingerprint"] != protocol["protocol_fingerprint"] or raw_manifest["run_manifest_fingerprint"] != manifest["manifest_fingerprint"]:
        raise RuntimeError("request-manifest linkage mismatch")
    if file_sha256(REQUESTS) != raw_manifest["requests_sha256"] or file_sha256(AUDIT) != raw_manifest["retrieval_audit_sha256"]:
        raise RuntimeError("request or audit file hash mismatch")

    expected_keys = [row["logical_key"] for row in manifest["pairs"]]
    request_keys = [logical_key(row) for row in requests]
    audit_keys = [row["logical_key"] for row in audits]
    if request_keys != expected_keys or audit_keys != expected_keys or len(set(request_keys)) != len(request_keys):
        raise RuntimeError("coverage, ordering, or logical-key uniqueness failure")
    request_pair_map = {row["logical_key"]: row for row in raw_manifest["pairs"]}
    if set(request_pair_map) != set(expected_keys):
        raise RuntimeError("raw request manifest coverage mismatch")
    for request in requests:
        key = logical_key(request)
        if heldout_request_fingerprint_v1_1(request) != request_pair_map[key]["heldout_request_fingerprint_v1_1"]:
            raise RuntimeError(f"request fingerprint mismatch: {key}")

    forbidden_keys = {"gold", "gold_answer", "gold_answers", "gold_evidence", "gold_evidence_spans", "reference_answer", "evidence_labels"}
    observed_keys: set[str] = set()
    def collect_keys(value: object) -> None:
        if isinstance(value, dict):
            observed_keys.update(str(key).lower() for key in value)
            for item in value.values():
                collect_keys(item)
        elif isinstance(value, list):
            for item in value:
                collect_keys(item)
    collect_keys(requests)
    collect_keys(audits)
    forbidden_found = sorted(forbidden_keys & observed_keys)
    if forbidden_found:
        raise RuntimeError(f"gold-like fields found in generation artifacts: {forbidden_found}")

    audit_map = {row["logical_key"]: row for row in audits}
    required_audit = {
        "configuration_fingerprint", "native_final_k", "native_context_tokens", "native_evidence",
        "supplied_evidence", "supplied_context_tokens", "packing_exclusions",
        "packing_truncated_chunk_count", "packing_skipped_complete_chunk_count", "stop_reason",
    }
    k_values: dict[str, set[int]] = defaultdict(set)
    system_counts = Counter()
    stop_reasons: dict[str, Counter] = defaultdict(Counter)
    max_context = 0
    requests_over_limit: list[str] = []
    packing_truncations = 0
    skipped_chunks = 0
    configuration_fingerprints: dict[str, set[str]] = defaultdict(set)
    for request in requests:
        key = logical_key(request)
        audit = audit_map[key]
        missing = required_audit - set(audit)
        if missing:
            raise RuntimeError(f"audit fields missing for {key}: {sorted(missing)}")
        supplied_request = [{"evidence_id": item["evidence_id"], "text": item["text"]} for item in audit["supplied_evidence"]]
        if request["evidence"] != supplied_request:
            raise RuntimeError(f"supplied evidence mismatch: {key}")
        if audit["supplied_context_tokens"] != sum(item["tokens"] for item in audit["supplied_evidence"]):
            raise RuntimeError(f"supplied token accounting mismatch: {key}")
        if audit["native_final_k"] != len(audit["native_evidence"]):
            raise RuntimeError(f"native final-k mismatch: {key}")
        if audit["packing_skipped_complete_chunk_count"] != len(audit["packing_exclusions"]):
            raise RuntimeError(f"packing exclusion count mismatch: {key}")
        max_context = max(max_context, int(audit["supplied_context_tokens"]))
        if audit["supplied_context_tokens"] > protocol["retrieval_freeze"]["common_generation_evidence_cap_tokens"]:
            requests_over_limit.append(key)
        packing_truncations += int(audit["packing_truncated_chunk_count"])
        skipped_chunks += int(audit["packing_skipped_complete_chunk_count"])
        system = request["system"]
        system_counts[system] += 1
        k_values[system].add(int(audit["native_final_k"]))
        stop_reasons[system][str(audit["stop_reason"])] += 1
        configuration_fingerprints[system].add(str(audit["configuration_fingerprint"]))
    if packing_truncations != 0 or requests_over_limit:
        raise RuntimeError("packing violated the frozen complete-chunk/cap policy")
    if len(k_values["sar_rag_v1"]) <= 1 or len(k_values["adaptive_single_view_v2"]) <= 1:
        raise RuntimeError("dynamic final k was not retained for both adaptive systems")
    if any(len(values) != 1 for values in configuration_fingerprints.values()):
        raise RuntimeError("a system used more than one configuration fingerprint")

    preflight_total = sum(int(row["preflight_input_tokens"]) for row in raw_manifest["pairs"])
    conservative_input_total = sum(int(row["conservative_billed_input_tokens"]) for row in raw_manifest["pairs"])
    input_limit = 2800
    input_limit_excesses = [
        {"logical_key": row["logical_key"], "tokens": row["conservative_billed_input_tokens"]}
        for row in raw_manifest["pairs"] if int(row["conservative_billed_input_tokens"]) > input_limit
    ]
    if input_limit_excesses:
        raise RuntimeError("one or more requests exceed the declared input reserve")
    initial_attempts = len(requests)
    maximum_attempts = int(manifest["maximum_total_api_attempts_including_retries"])
    max_output = int(protocol["generation"]["maximum_completion_tokens"])
    input_rate = Decimal(str(protocol["cost_and_attempt_plan"]["pricing"]["input_per_million"]))
    output_rate = Decimal(str(protocol["cost_and_attempt_plan"]["pricing"]["output_per_million"]))
    million = Decimal(1_000_000)
    conservative_initial = (Decimal(conservative_input_total) * input_rate + Decimal(initial_attempts * max_output) * output_rate) / million
    retry_multiplier = Decimal(maximum_attempts) / Decimal(initial_attempts)
    conservative_maximum = conservative_initial * retry_multiplier
    smoke_mean_output = Decimal("503.5")
    estimated_initial = (Decimal(preflight_total) * input_rate + Decimal(initial_attempts) * smoke_mean_output * output_rate) / million

    sealed = deepcopy(raw_manifest)
    sealed.pop("request_manifest_fingerprint", None)
    sealed.update({
        "status": "FROZEN_SEALED_REQUESTS_ZERO_API_CALLS",
        "source_request_manifest_fingerprint": raw_manifest["request_manifest_fingerprint"],
        "total_preflight_input_tokens": preflight_total,
        "total_conservative_billed_input_tokens": conservative_input_total,
        "total_reserved_output_tokens_initial": initial_attempts * max_output,
        "total_reserved_output_tokens_maximum": maximum_attempts * max_output,
        "estimated_initial_cost_usd_from_actual_requests_and_smoke_mean_output": float(estimated_initial),
        "conservative_initial_cost_usd_from_sealed_requests": float(conservative_initial),
        "maximum_cost_usd_from_sealed_requests": float(conservative_maximum),
        "declared_conservative_input_limit_tokens": input_limit,
        "requests_exceeding_declared_limits": [],
        "account_specific_rate_limits_verified": False,
    })
    sealed["request_manifest_fingerprint"] = object_sha256(sealed)
    SEALED_REQUEST_MANIFEST.write_text(json.dumps(sealed, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = {
        "status": "PASS",
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "run_manifest_fingerprint": manifest["manifest_fingerprint"],
        "source_request_manifest_fingerprint": raw_manifest["request_manifest_fingerprint"],
        "sealed_request_manifest_fingerprint": sealed["request_manifest_fingerprint"],
        "checks": {
            "protocol_fingerprint_valid": protocol_valid,
            "manifest_fingerprint_valid": manifest_valid,
            "raw_request_manifest_fingerprint_valid": raw_manifest_valid,
            "frozen_linkage_valid": True,
            "requests_file_hash_valid": True,
            "retrieval_audit_file_hash_valid": True,
            "complete_coverage": len(requests) == manifest["pair_count"],
            "unique_pair_keys": len(set(request_keys)) == len(request_keys),
            "exact_manifest_order": request_keys == expected_keys,
            "all_request_fingerprints_valid": True,
            "generation_artifacts_gold_free": not forbidden_found,
            "required_retrieval_audit_fields_present": True,
            "request_evidence_matches_supplied_evidence": True,
            "dynamic_k_retained_sar_rag_v1": len(k_values["sar_rag_v1"]) > 1,
            "dynamic_k_retained_adaptive_single_view_v2": len(k_values["adaptive_single_view_v2"]) > 1,
            "no_chunk_truncation": packing_truncations == 0,
            "all_declared_token_limits_respected": not input_limit_excesses and not requests_over_limit,
        },
        "pair_count": len(requests),
        "system_counts": dict(sorted(system_counts.items())),
        "native_final_k_values": {key: sorted(values) for key, values in sorted(k_values.items())},
        "configuration_fingerprints": {key: sorted(values) for key, values in sorted(configuration_fingerprints.items())},
        "stop_reason_counts": {key: dict(sorted(values.items())) for key, values in sorted(stop_reasons.items())},
        "maximum_supplied_context_tokens": max_context,
        "packing_truncated_chunk_count": packing_truncations,
        "packing_skipped_complete_chunk_count": skipped_chunks,
        "input_reserve": {
            "preflight_total": preflight_total,
            "conservative_billed_total": conservative_input_total,
            "maximum_preflight_per_request": raw_manifest["maximum_preflight_input_tokens"],
            "maximum_conservative_per_request": raw_manifest["maximum_conservative_billed_input_tokens"],
            "declared_limit_per_request": input_limit,
            "exceeding_requests": input_limit_excesses,
        },
        "cost": {
            "estimated_initial_usd": float(estimated_initial),
            "conservative_initial_usd": float(conservative_initial),
            "conservative_maximum_usd": float(conservative_maximum),
            "estimated_not_invoiced": True,
        },
        "gold_labels_accessed": False,
        "api_calls": 0,
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    min_seconds = int(protocol["failure_policy"]["rate_limit"]["minimum_seconds_between_attempt_starts"])
    initial_hours = initial_attempts * min_seconds / 3600
    maximum_hours = maximum_attempts * min_seconds / 3600
    COST_PLAN.write_text(f"""# Sealed-request calls, cost, and runtime\n\nPricing remains the frozen official `openai/gpt-oss-120b` rate: $0.15/M uncached input tokens and $0.60/M output tokens. All figures below are estimates, not actual invoiced charges.\n\n| Quantity | Sealed value |\n|---|---:|\n| Requests / initial attempts | {initial_attempts:,} |\n| Maximum attempts (3 per pair) | {maximum_attempts:,} |\n| Actual preflight input tokens, initial | {preflight_total:,} |\n| Conservative billed input reserve, initial | {conservative_input_total:,} |\n| Reserved output tokens, initial | {initial_attempts * max_output:,} |\n| Reserved output tokens, maximum | {maximum_attempts * max_output:,} |\n| Smoke-mean planning estimate, initial | ${float(estimated_initial):.8f} |\n| Conservative initial reserve | ${float(conservative_initial):.8f} |\n| Final conservative authorization ceiling | ${float(conservative_maximum):.8f} |\n\nNo sealed request exceeds the 2,800-token conservative input limit or the 1,600-token supplied-context/completion limits. The runner checks the remaining attempt and exact per-request reserved-cost allowance before every call.\n\nThe frozen serial pacing of {min_seconds} seconds between attempt starts gives a no-stall lower bound of {initial_hours:.2f} hours ({initial_hours / 24:.2f} days) for the initial round and {maximum_hours:.2f} hours ({maximum_hours / 24:.2f} days) at the retry ceiling. Account RPM/TPM/RPD/TPD limits remain unknown and may increase elapsed time; they cannot reduce this serial pacing lower bound.\n""", encoding="utf-8")

    cmd_prefix = ".\\.venv\\Scripts\\python.exe"
    COMMANDS.write_text(f"""# Sealed held-out v1.1 commands\n\nAll paths are relative to the workspace root. The generation command remains blocked unless both `--execute-paid` and a separately authorized, fingerprint-exact authorization file are supplied.\n\n## Revalidate offline package\n\n```powershell\n& '{cmd_prefix}' tools/seal_heldout_requests_v1_1.py\n& '{cmd_prefix}' tools/validate_sealed_heldout_execution_v1_1.py\n& '{cmd_prefix}' -m pytest tests/test_heldout_v1_1.py -q\n```\n\nThe seal command is intentionally create-once and will refuse to overwrite sealed artifacts.\n\n## Paid generation (not authorized by this document)\n\n```powershell\n& '{cmd_prefix}' tools/run_final_heldout_v1_1.py `\n  --protocol output/final_evaluation/heldout_v1_1/FINAL_EVALUATION_PROTOCOL_AMENDMENT_V1_1.json `\n  --manifest output/final_evaluation/heldout_v1_1/FROZEN_QUESTION_SYSTEM_MANIFEST.json `\n  --requests output/final_evaluation/heldout_v1_1/REQUESTS.jsonl `\n  --request-manifest output/final_evaluation/heldout_v1_1/REQUEST_MANIFEST_SEALED.json `\n  --authorization-file output/final_evaluation/heldout_v1_1/AUTHORIZATION.json `\n  --checkpoint-dir output/final_evaluation/heldout_v1_1/generation_checkpoint `\n  --execute-paid\n```\n\n## Scoring (separately gated; do not run during generation authorization)\n\n```powershell\n& '{cmd_prefix}' tools/score_final_heldout_v1_1.py --help\n```\n""", encoding="utf-8")

    authorization = {
        "status": "DISABLED_PROPOSAL_NOT_AUTHORIZATION",
        "authorized_stage": "heldout_generation_v1_1_only",
        "user_authorized": False,
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "run_manifest_fingerprint": manifest["manifest_fingerprint"],
        "request_manifest_fingerprint": sealed["request_manifest_fingerprint"],
        "maximum_total_api_attempts_including_retries": maximum_attempts,
        "maximum_attempts_per_pair_including_first": manifest["maximum_attempts_per_pair_including_first"],
        "maximum_cost_usd": float(conservative_maximum),
        "scope_note": "No scoring, judge calls, configuration changes, calls outside the sealed pairs, or reuse of unused allowance.",
    }
    AUTHORIZATION.write_text(json.dumps(authorization, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LIMITS.write_text(f"""# Account-specific Groq limits still required\n\nThe Groq console limits page was checked read-only without a generation call, but the available browser session was signed out. No account-specific limit could be verified.\n\nBefore paid authorization, provide the effective limits for the exact organization/project and model `openai/gpt-oss-120b`:\n\n- requests per minute (RPM)\n- requests per day (RPD)\n- tokens per minute (TPM), including whether input/output have separate limits\n- tokens per day (TPD), including whether input/output have separate limits\n- the applicable plan/tier and any project-level overrides\n- whether failed or rate-limited attempts count against each limit\n\nThe executable remains serial with a frozen {min_seconds}-second minimum interval. Limits are needed to estimate quota stalls and confirm that the initial {initial_attempts:,} requests and maximum {maximum_attempts:,} attempts can complete within the intended calendar window. No generation call will be used to discover these values.\n""", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "pair_count": len(requests),
        "request_manifest_fingerprint": sealed["request_manifest_fingerprint"],
        "conservative_maximum_cost_usd": float(conservative_maximum),
        "requests_exceeding_limits": 0,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
