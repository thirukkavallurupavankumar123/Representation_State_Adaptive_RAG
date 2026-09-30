"""Exercise the sealed held-out runner with a zero-network fake answerer."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from sar_rag.final_evaluation import object_sha256  # noqa: E402
from sar_rag.heldout_v1_1 import HeldoutV11Runner  # noqa: E402
from sar_rag.smoke_recovery import AttemptOutcome, load_jsonl  # noqa: E402
from run_final_heldout_v1_1 import verify_authorization  # noqa: E402


OUT = ROOT / "output/final_evaluation/heldout_v1_1"
PROTOCOL = OUT / "FINAL_EVALUATION_PROTOCOL_AMENDMENT_V1_1.json"
MANIFEST = OUT / "FROZEN_QUESTION_SYSTEM_MANIFEST.json"
REQUESTS = OUT / "REQUESTS.jsonl"
REQUEST_MANIFEST = OUT / "REQUEST_MANIFEST_SEALED.json"
DISABLED_AUTH = OUT / "PROPOSED_AUTHORIZATION_SEALED_REQUESTS_DISABLED.json"
REPORT = OUT / "SEALED_EXECUTION_DRY_RUN.json"


class FakeAnswerer:
    def __init__(self) -> None:
        self.calls = 0

    def attempt(self, request: dict) -> AttemptOutcome:
        self.calls += 1
        payload = {
            "answerability": "INSUFFICIENT",
            "claims": [],
            "missing_information": "Fake-answerer dry run; no model was called.",
        }
        return AttemptOutcome(
            status="success", payload=payload, raw_response=json.dumps(payload),
            input_tokens=1, output_tokens=1, latency_ms=0,
        )


def main() -> int:
    if REPORT.exists():
        raise FileExistsError(f"refusing to overwrite dry-run report: {REPORT}")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    request_manifest = json.loads(REQUEST_MANIFEST.read_text(encoding="utf-8"))
    requests = load_jsonl(REQUESTS)

    disabled_authorization_rejected = False
    try:
        verify_authorization(DISABLED_AUTH, protocol, manifest, request_manifest)
    except SystemExit:
        disabled_authorization_rejected = True
    if not disabled_authorization_rejected:
        raise RuntimeError("disabled authorization proposal was accepted")

    effective_maximum_cost = min(
        float(manifest["maximum_cost_usd"]),
        float(request_manifest["maximum_cost_usd_from_sealed_requests"]),
    )
    matching_authorization = {
        "status": "EXPLICIT_USER_AUTHORIZATION_RECORDED",
        "authorized_stage": "heldout_generation_v1_1_only",
        "user_authorized": True,
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "run_manifest_fingerprint": manifest["manifest_fingerprint"],
        "request_manifest_fingerprint": request_manifest["request_manifest_fingerprint"],
        "maximum_total_api_attempts_including_retries": manifest["maximum_total_api_attempts_including_retries"],
        "maximum_attempts_per_pair_including_first": manifest["maximum_attempts_per_pair_including_first"],
        "maximum_cost_usd": effective_maximum_cost,
    }

    temp_root = Path(tempfile.mkdtemp(prefix="heldout_v1_1_dry_run_"))
    try:
        authorization_path = temp_root / "AUTHORIZATION.json"
        authorization_path.write_text(json.dumps(matching_authorization, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        verify_authorization(authorization_path, protocol, manifest, request_manifest)
        checkpoint = temp_root / "checkpoint"
        fake = FakeAnswerer()
        runner = HeldoutV11Runner(
            checkpoint, protocol, manifest, request_manifest,
            protocol_path=PROTOCOL, manifest_path=MANIFEST, request_manifest_path=REQUEST_MANIFEST,
        )
        first = runner.run(requests, fake)
        first_calls = fake.calls
        resume_fake = FakeAnswerer()
        resumed = HeldoutV11Runner(
            checkpoint, protocol, manifest, request_manifest,
            protocol_path=PROTOCOL, manifest_path=MANIFEST, request_manifest_path=REQUEST_MANIFEST,
        ).run(requests, resume_fake)

        modified_protocol = deepcopy(protocol)
        modified_protocol["dry_run_forbidden_change"] = True
        modified_protocol["protocol_fingerprint"] = object_sha256({
            key: value for key, value in modified_protocol.items() if key != "protocol_fingerprint"
        })
        checkpoint_drift_rejected = False
        try:
            HeldoutV11Runner(checkpoint, modified_protocol, manifest, request_manifest)
        except ValueError:
            checkpoint_drift_rejected = True

        drifted_requests = deepcopy(requests)
        drifted_requests[0]["question"] += " DRIFT"
        request_drift_rejected = False
        drift_fake = FakeAnswerer()
        try:
            HeldoutV11Runner(temp_root / "drift", protocol, manifest, request_manifest).run(drifted_requests, drift_fake)
        except ValueError:
            request_drift_rejected = drift_fake.calls == 0

        report = {
            "status": "PASS",
            "sealed_request_manifest_fingerprint": request_manifest["request_manifest_fingerprint"],
            "pair_count": len(requests),
            "fake_first_run": first,
            "fake_attempt_calls": first_calls,
            "resume": resumed,
            "resume_fake_calls": resume_fake.calls,
            "checks": {
                "disabled_authorization_rejected": disabled_authorization_rejected,
                "matching_hypothetical_authorization_accepted_locally": True,
                "all_sealed_requests_validated_first_attempt": first["first_attempt_valid_pairs"] == len(requests),
                "no_duplicate_calls_after_resume": resume_fake.calls == 0 and resumed["new_attempts"] == 0,
                "checkpoint_fingerprint_drift_rejected": checkpoint_drift_rejected,
                "request_content_drift_rejected_before_call": request_drift_rejected,
                "attempt_ceiling_respected": first["cumulative_attempts"] == len(requests),
            },
            "gold_labels_accessed": False,
            "api_calls": 0,
        }
        if not all(report["checks"].values()):
            raise RuntimeError(f"one or more sealed execution dry-run checks failed: {report['checks']}")
        REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
