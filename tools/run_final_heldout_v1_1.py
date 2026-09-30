"""Run the separately authorized frozen held-out v1.1 generation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sar_rag.heldout_v1_1 import HeldoutV11Runner  # noqa: E402
from sar_rag.smoke_recovery import load_jsonl  # noqa: E402
from run_smoke_generation_v1_1 import GroqV11Answerer  # noqa: E402


def verify_authorization(path: Path | None, protocol: dict, manifest: dict,
                         request_manifest: dict, free_quota_plan: dict | None = None) -> None:
    if path is None or not path.is_file():
        raise SystemExit("HELD-OUT GENERATION BLOCKED: matching authorization file is absent")
    actual = json.loads(path.read_text(encoding="utf-8"))
    effective_maximum_cost = min(
        float(manifest["maximum_cost_usd"]),
        float(request_manifest.get(
            "maximum_cost_usd_from_sealed_requests",
            manifest["maximum_cost_usd"],
        )),
    )
    if free_quota_plan is None:
        raise SystemExit("HELD-OUT GENERATION BLOCKED: frozen free-quota plan is required")
    expected = {
        "status": "EXPLICIT_USER_AUTHORIZATION_RECORDED",
        "authorized_stage": "heldout_generation_v1_1_free_quota_only",
        "user_authorized": True,
        "free_quota_only": True,
        "billing_upgrade_or_credit_purchase_authorized": False,
        "maximum_billed_cost_usd": 0.0,
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "run_manifest_fingerprint": manifest["manifest_fingerprint"],
        "request_manifest_fingerprint": request_manifest["request_manifest_fingerprint"],
        "quota_plan_fingerprint": free_quota_plan["quota_plan_fingerprint"],
        "maximum_total_api_attempts_including_retries": manifest["maximum_total_api_attempts_including_retries"],
        "maximum_attempts_per_pair_including_first": manifest["maximum_attempts_per_pair_including_first"],
        "counterfactual_list_price_cost_ceiling_usd": effective_maximum_cost,
    }
    if actual != expected:
        raise SystemExit("HELD-OUT GENERATION BLOCKED: authorization content/fingerprint mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--request-manifest", type=Path, required=True)
    parser.add_argument("--free-quota-plan", type=Path, required=True)
    parser.add_argument("--authorization-file", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--execute-paid", action="store_true")
    args = parser.parse_args()
    if not args.execute_paid:
        raise SystemExit("HELD-OUT GENERATION BLOCKED: --execute-paid was not provided")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    request_manifest = json.loads(args.request_manifest.read_text(encoding="utf-8"))
    free_quota_plan = json.loads(args.free_quota_plan.read_text(encoding="utf-8"))
    quota_body = {key: value for key, value in free_quota_plan.items() if key != "quota_plan_fingerprint"}
    from sar_rag.final_evaluation import object_sha256
    if object_sha256(quota_body) != free_quota_plan.get("quota_plan_fingerprint"):
        raise SystemExit("HELD-OUT GENERATION BLOCKED: invalid free-quota plan fingerprint")
    verify_authorization(args.authorization_file, protocol, manifest, request_manifest, free_quota_plan)
    if protocol["separation_and_scoring_gate"]["heldout_generation_authorized"]:
        raise SystemExit("frozen protocol must remain an authorization-neutral specification")
    from sar_rag.environment import load_project_env
    load_project_env()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("HELD-OUT GENERATION BLOCKED: GROQ_API_KEY unavailable")
    requests = load_jsonl(args.requests)
    runner = HeldoutV11Runner(
        args.checkpoint_dir, protocol, manifest, request_manifest,
        protocol_path=args.protocol, manifest_path=args.manifest,
        request_manifest_path=args.request_manifest,
        free_quota_plan=free_quota_plan,
    )
    summary = runner.run(
        requests,
        GroqV11Answerer(
            api_key, timeout_seconds=90,
            minimum_interval_seconds=float(protocol["failure_policy"]["rate_limit"]["minimum_seconds_between_attempt_starts"]),
        ),
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
