"""Dry-run the sealed free-quota controller without network access."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from sar_rag.heldout_v1_1 import HeldoutV11Runner  # noqa: E402
from sar_rag.smoke_recovery import AttemptOutcome, load_jsonl  # noqa: E402
from run_final_heldout_v1_1 import verify_authorization  # noqa: E402

OUT = ROOT / "output/final_evaluation/heldout_v1_1"
REPORT = OUT / "FREE_QUOTA_DRY_RUN.json"


class Fake:
    def __init__(self, clock=None) -> None:
        self.calls = 0
        self.clock = clock
    def attempt(self, request: dict) -> AttemptOutcome:
        self.calls += 1
        if self.clock is not None:
            self.clock[0] += timedelta(seconds=30)
        payload = {"answerability": "INSUFFICIENT", "claims": [], "missing_information": "offline fake"}
        return AttemptOutcome(status="success", payload=payload, raw_response=json.dumps(payload), input_tokens=1, output_tokens=1, latency_ms=0, http_status=200)


def main() -> int:
    if REPORT.exists():
        raise FileExistsError(f"refusing to overwrite: {REPORT}")
    protocol_path = OUT / "FINAL_EVALUATION_PROTOCOL_AMENDMENT_V1_1.json"
    manifest_path = OUT / "FROZEN_QUESTION_SYSTEM_MANIFEST.json"
    request_manifest_path = OUT / "REQUEST_MANIFEST_SEALED.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    request_manifest = json.loads(request_manifest_path.read_text(encoding="utf-8"))
    plan = json.loads((OUT / "FREE_QUOTA_EXECUTION_PLAN.json").read_text(encoding="utf-8"))
    disabled_path = OUT / "PROPOSED_FREE_QUOTA_AUTHORIZATION_DISABLED.json"
    requests = load_jsonl(OUT / "REQUESTS.jsonl")
    disabled_rejected = False
    try:
        verify_authorization(disabled_path, protocol, manifest, request_manifest, plan)
    except SystemExit:
        disabled_rejected = True
    authorization = {
        "status": "EXPLICIT_USER_AUTHORIZATION_RECORDED",
        "authorized_stage": "heldout_generation_v1_1_free_quota_only",
        "user_authorized": True,
        "free_quota_only": True,
        "billing_upgrade_or_credit_purchase_authorized": False,
        "maximum_billed_cost_usd": 0.0,
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "run_manifest_fingerprint": manifest["manifest_fingerprint"],
        "request_manifest_fingerprint": request_manifest["request_manifest_fingerprint"],
        "quota_plan_fingerprint": plan["quota_plan_fingerprint"],
        "maximum_total_api_attempts_including_retries": manifest["maximum_total_api_attempts_including_retries"],
        "maximum_attempts_per_pair_including_first": manifest["maximum_attempts_per_pair_including_first"],
        "counterfactual_list_price_cost_ceiling_usd": request_manifest["maximum_cost_usd_from_sealed_requests"],
    }
    temp = Path(tempfile.mkdtemp(prefix="free_quota_dry_run_"))
    try:
        auth = temp / "AUTH.json"
        auth.write_text(json.dumps(authorization, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        verify_authorization(auth, protocol, manifest, request_manifest, plan)
        clock = [datetime(2026, 9, 27, tzinfo=timezone.utc)]
        fake = Fake(clock)
        pauses = 0
        last = None
        while True:
            runner = HeldoutV11Runner(temp / "checkpoint", protocol, manifest, request_manifest,
                                      protocol_path=protocol_path, manifest_path=manifest_path,
                                      request_manifest_path=request_manifest_path,
                                      free_quota_plan=plan, now_fn=lambda: clock[0])
            last = runner.run(requests, fake)
            if last["eventually_valid_pairs"] == len(requests):
                break
            state = json.loads((temp / "checkpoint/FREE_QUOTA_STATE.json").read_text(encoding="utf-8"))
            clock[0] = datetime.fromisoformat(state["resume_not_before_utc"]) + timedelta(seconds=1)
            pauses += 1
            if pauses > 30:
                raise RuntimeError("unexpected number of free-quota windows")
        resume_fake = Fake()
        resumed = HeldoutV11Runner(temp / "checkpoint", protocol, manifest, request_manifest,
                                   free_quota_plan=plan, now_fn=lambda: clock[0]).run(requests, resume_fake)
        report = {
            "status": "PASS",
            "quota_plan_fingerprint": plan["quota_plan_fingerprint"],
            "pair_count": len(requests),
            "fake_calls": fake.calls,
            "quota_pauses": pauses,
            "eventually_valid_pairs": last["eventually_valid_pairs"],
            "resume_duplicate_calls": resume_fake.calls,
            "checks": {
                "disabled_authorization_rejected": disabled_rejected,
                "matching_hypothetical_authorization_accepted": True,
                "quota_windows_checkpointed": pauses > 0,
                "all_pairs_eventually_completed": last["eventually_valid_pairs"] == len(requests),
                "no_duplicate_calls_after_completion": resumed["new_attempts"] == 0 and resume_fake.calls == 0,
                "no_network_calls": True,
            },
            "api_calls": 0,
        }
        if not all(report["checks"].values()):
            raise RuntimeError(report)
        REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        shutil.rmtree(temp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
