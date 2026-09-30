"""Offline authorization and integrity audit before any paid v1.1 request."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from sar_rag.environment import load_project_env  # noqa: E402
from sar_rag.final_evaluation import file_sha256, object_sha256  # noqa: E402
from tools.run_smoke_generation_v1_1 import verify_authorization  # noqa: E402


def main() -> int:
    output = ROOT / "output/final_evaluation"
    base = output / "generation_v1_1"
    protocol_path = base / "PROTOCOL.json"
    manifest_path = base / "RUN_MANIFEST.json"
    authorization_path = base / "PAID_AUTHORIZATION.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_authorization(authorization_path, protocol, manifest)
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
    checkpoint = base / "checkpoint"
    prior_attempts = 0
    if (checkpoint / "API_ATTEMPTS.jsonl").exists():
        prior_attempts = len([
            line for line in (checkpoint / "API_ATTEMPTS.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ])
    load_project_env()
    report = {
        "status": "VERIFIED_READY_FOR_AUTHORIZED_V1_1_SMOKE",
        "protocol_fingerprint_valid": protocol_fingerprint_valid,
        "manifest_fingerprint_valid": manifest_fingerprint_valid,
        "protocol_manifest_link_valid": manifest["protocol_fingerprint"] == protocol["protocol_fingerprint"],
        "preserved_artifact_hashes_match": preserved_hashes_match,
        "authorization_matches": True,
        "eligible_pairs": manifest["pair_count"],
        "maximum_attempts_per_pair": manifest["maximum_attempts_per_pair_including_first"],
        "maximum_total_attempts": manifest["maximum_total_api_attempts_including_retries"],
        "prior_attempts": prior_attempts,
        "remaining_attempt_allowance": manifest["maximum_total_api_attempts_including_retries"] - prior_attempts,
        "maximum_cost_usd": manifest["maximum_cost_usd"],
        "remaining_conservative_cost_allowance_usd": manifest["maximum_cost_usd"],
        "api_key_available_but_not_recorded": bool(os.environ.get("GROQ_API_KEY")),
        "heldout_checkpoint_exists": (output / "paid_heldout_checkpoint").exists(),
    }
    if not all((
        protocol_fingerprint_valid,
        manifest_fingerprint_valid,
        report["protocol_manifest_link_valid"],
        preserved_hashes_match,
        report["eligible_pairs"] == 48,
        report["maximum_attempts_per_pair"] == 3,
        report["maximum_total_attempts"] == 144,
        prior_attempts == 0,
        report["api_key_available_but_not_recorded"],
        not report["heldout_checkpoint_exists"],
    )):
        raise RuntimeError("v1.1 paid pre-execution validation failed")
    destination = base / "PREEXECUTION_VALIDATION.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
