"""Freeze a disabled free-plan quota execution proposal; never calls Groq."""

from __future__ import annotations

from decimal import Decimal, ROUND_CEILING
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sar_rag.final_evaluation import object_sha256  # noqa: E402

OUT = ROOT / "output/final_evaluation/heldout_v1_1"


def ceiling(value: Decimal) -> int:
    return int(value.to_integral_value(rounding=ROUND_CEILING))


def main() -> int:
    plan_path = OUT / "FREE_QUOTA_EXECUTION_PLAN.json"
    auth_path = OUT / "PROPOSED_FREE_QUOTA_AUTHORIZATION_DISABLED.json"
    report_path = OUT / "FREE_QUOTA_RUNTIME_ESTIMATE.md"
    commands_path = OUT / "FREE_QUOTA_EXECUTION_COMMANDS.md"
    for path in (plan_path, auth_path, report_path, commands_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite free-quota artifact: {path}")
    protocol = json.loads((OUT / "FINAL_EVALUATION_PROTOCOL_AMENDMENT_V1_1.json").read_text(encoding="utf-8"))
    manifest = json.loads((OUT / "FROZEN_QUESTION_SYSTEM_MANIFEST.json").read_text(encoding="utf-8"))
    requests = json.loads((OUT / "REQUEST_MANIFEST_SEALED.json").read_text(encoding="utf-8"))
    limits = {"rpm": 30, "rpd": 1000, "tpm": 8000, "tpd": 200000}
    initial = int(requests["pair_count"])
    maximum = int(manifest["maximum_total_api_attempts_including_retries"])
    expected_tokens = Decimal(int(requests["total_preflight_input_tokens"])) + Decimal("503.5") * initial
    conservative_tokens = Decimal(int(requests["total_conservative_billed_input_tokens"])) + Decimal(int(requests["total_reserved_output_tokens_initial"]))
    multiplier = Decimal(maximum) / Decimal(initial)
    plan = {
        "status": "FROZEN_DISABLED_AWAITING_EXPLICIT_FREE_QUOTA_AUTHORIZATION",
        "version": "heldout-v1.1-free-quota-1.0.0",
        "model": "openai/gpt-oss-120b",
        "account_plan": "free",
        "account_plan_source": "user declaration",
        "published_limit_source": "https://console.groq.com/docs/rate-limits",
        "published_limits_verified_date": "2026-09-27",
        "limits": limits,
        "organization_override_verified": False,
        "organization_override_note": "The Groq console was signed out; published free-plan limits are enforced conservatively. Response headers are captured once execution is separately authorized.",
        "enforcement": {
            "minute_window_seconds": 60,
            "daily_window_seconds": 86400,
            "window_model": "local rolling windows; conservative when provider reset boundary is unavailable",
            "pre_call_reservation": "sealed conservative input tokens plus 1,600 maximum output tokens",
            "post_call_accounting": "record actual provider usage when present; retain conservative reservation when usage is absent",
            "quota_exhaustion": "checkpoint and exit with resume_not_before_utc",
            "http_429": "record once, checkpoint, pause immediately, honor Retry-After when supplied; never retry in the same invocation",
            "resume": "rerun the exact command after resume_not_before_utc; answer/request fingerprints prevent duplicate validated calls",
        },
        "billing_policy": {
            "free_quota_only": True,
            "enable_billing": False,
            "upgrade_plan": False,
            "purchase_credits": False,
            "maximum_billed_cost_usd": 0.0,
        },
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "run_manifest_fingerprint": manifest["manifest_fingerprint"],
        "request_manifest_fingerprint": requests["request_manifest_fingerprint"],
        "initial_attempts": initial,
        "maximum_attempts_including_retries": maximum,
        "expected_initial_tokens": int(expected_tokens),
        "conservative_initial_tokens": int(conservative_tokens),
        "expected_initial_quota_days": ceiling(expected_tokens / limits["tpd"]),
        "conservative_initial_quota_days": ceiling(conservative_tokens / limits["tpd"]),
        "expected_maximum_quota_days": ceiling(expected_tokens * multiplier / limits["tpd"]),
        "conservative_maximum_quota_days": ceiling(conservative_tokens * multiplier / limits["tpd"]),
        "generation_enabled": False,
        "api_calls": 0,
    }
    plan["quota_plan_fingerprint"] = object_sha256(plan)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    disabled = {
        "status": "DISABLED_PROPOSAL_NOT_AUTHORIZATION",
        "authorized_stage": "heldout_generation_v1_1_free_quota_only",
        "user_authorized": False,
        "free_quota_only": True,
        "billing_upgrade_or_credit_purchase_authorized": False,
        "maximum_billed_cost_usd": 0.0,
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "run_manifest_fingerprint": manifest["manifest_fingerprint"],
        "request_manifest_fingerprint": requests["request_manifest_fingerprint"],
        "quota_plan_fingerprint": plan["quota_plan_fingerprint"],
        "maximum_total_api_attempts_including_retries": maximum,
        "maximum_attempts_per_pair_including_first": manifest["maximum_attempts_per_pair_including_first"],
        "counterfactual_list_price_cost_ceiling_usd": requests["maximum_cost_usd_from_sealed_requests"],
    }
    auth_path.write_text(json.dumps(disabled, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(f"""# Free-quota runtime estimate\n\nPublished free limits for `openai/gpt-oss-120b`: 30 RPM, 1,000 RPD, 8,000 TPM, and 200,000 TPD. Groq says limits are organization-level and exceptions may exist; the private console override could not be checked because the available browser was signed out.\n\n| Scenario | Tokens | TPD-dominated quota windows |\n|---|---:|---:|\n| Initial, smoke-mean output | {int(expected_tokens):,} | {plan['expected_initial_quota_days']} days |\n| Initial, conservative 1,600 output | {int(conservative_tokens):,} | {plan['conservative_initial_quota_days']} days |\n| Three-attempt maximum, smoke-mean output | {int(expected_tokens * multiplier):,} | {plan['expected_maximum_quota_days']} days |\n| Three-attempt maximum, conservative | {int(conservative_tokens * multiplier):,} | {plan['conservative_maximum_quota_days']} days |\n\nTPD dominates RPD (19 initial request-days and 57 maximum request-days). Calendar completion cannot be shorter than the quota-window count and may be longer because quota is organization-wide, other usage consumes the same limits, actual reset boundaries are provider-controlled, and persistent failures can require retries. The runner uses conservative rolling 24-hour accounting when a provider daily reset is not exposed.\n\nGeneration remains disabled. Billing, plan upgrades, and credit purchases are prohibited; maximum authorized billed cost is $0.\n""", encoding="utf-8")
    commands_path.write_text("""# Free-quota held-out execution commands

The command remains blocked until `AUTHORIZATION_FREE_QUOTA.json` exactly matches the frozen free-quota authorization schema. Each invocation runs until completion or a quota pause. After a pause, read `generation_checkpoint/FREE_QUOTA_STATE.json` and rerun only after `resume_not_before_utc`.

```powershell
& '.\\.venv\\Scripts\\python.exe' tools/run_final_heldout_v1_1.py `
  --protocol output/final_evaluation/heldout_v1_1/FINAL_EVALUATION_PROTOCOL_AMENDMENT_V1_1.json `
  --manifest output/final_evaluation/heldout_v1_1/FROZEN_QUESTION_SYSTEM_MANIFEST.json `
  --requests output/final_evaluation/heldout_v1_1/REQUESTS.jsonl `
  --request-manifest output/final_evaluation/heldout_v1_1/REQUEST_MANIFEST_SEALED.json `
  --free-quota-plan output/final_evaluation/heldout_v1_1/FREE_QUOTA_EXECUTION_PLAN.json `
  --authorization-file output/final_evaluation/heldout_v1_1/AUTHORIZATION_FREE_QUOTA.json `
  --checkpoint-dir output/final_evaluation/heldout_v1_1/generation_checkpoint `
  --execute-paid
```

`--execute-paid` is the historical network-execution safety flag; the bound authorization requires free quota only, $0 maximum billed cost, and forbids billing, upgrades, or credit purchases.
""", encoding="utf-8")
    print(json.dumps({"status": "PASS", "quota_plan_fingerprint": plan["quota_plan_fingerprint"], "initial_expected_days": plan["expected_initial_quota_days"], "initial_conservative_days": plan["conservative_initial_quota_days"], "maximum_conservative_days": plan["conservative_maximum_quota_days"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
