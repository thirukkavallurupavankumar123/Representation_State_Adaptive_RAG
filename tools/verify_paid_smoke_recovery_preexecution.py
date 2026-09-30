"""Offline pre-execution audit for the separately authorized smoke recovery."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from sar_rag.final_evaluation import file_sha256  # noqa: E402
from sar_rag.environment import load_project_env  # noqa: E402
from sar_rag.smoke_recovery import load_jsonl  # noqa: E402
from tools.run_paid_smoke_recovery import verify_authorization  # noqa: E402


def main() -> int:
    output = ROOT / "output/final_evaluation"
    recovery = output / "recovery_v1"
    manifest_path = recovery / "RECOVERY_MANIFEST.json"
    authorization_path = recovery / "PAID_SMOKE_RECOVERY_AUTHORIZATION.json"
    checkpoint = recovery / "checkpoint"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_authorization(authorization_path, manifest)
    original_hashes_match = all(
        file_sha256(output / name) == expected
        for name, expected in manifest["original_integrity_sha256"].items()
    )
    meta = json.loads(
        (output / "paid_smoke_checkpoint/CHECKPOINT_META.json").read_text(encoding="utf-8")
    )
    generation = meta["generation"]
    generation_matches = (
        generation["model"] == "openai/gpt-oss-120b"
        and generation["decoding"] == {
            "max_output_tokens": 800,
            "reasoning_effort": "medium",
            "response_format": "strict JSON schema",
            "temperature": 0,
            "top_p": 1,
        }
    )
    attempts = load_jsonl(checkpoint / "API_ATTEMPTS.jsonl")
    reserved_cost = sum(
        float(row["authorized_worst_case_cost_usd"]) for row in attempts
    )
    load_project_env()
    report = {
        "status": "VERIFIED_READY_FOR_AUTHORIZED_PAID_RECOVERY",
        "manifest_sha256": file_sha256(manifest_path),
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "authorization_matches": True,
        "original_hashes_match": original_hashes_match,
        "generation_configuration_unchanged": generation_matches,
        "eligible_pair_count": len(manifest["pairs"]),
        "maximum_total_api_attempts_including_retries": manifest[
            "maximum_total_api_attempts_including_retries"
        ],
        "prior_recovery_attempts": len(attempts),
        "remaining_attempt_allowance": (
            manifest["maximum_total_api_attempts_including_retries"] - len(attempts)
        ),
        "maximum_cost_usd": manifest["maximum_cost_usd"],
        "reserved_worst_case_cost_usd": reserved_cost,
        "remaining_cost_allowance_usd": manifest["maximum_cost_usd"] - reserved_cost,
        "api_key_available_but_not_recorded": bool(os.environ.get("GROQ_API_KEY")),
        "held_out_evaluation_prohibited": True,
        "held_out_checkpoint_exists": (output / "paid_heldout_checkpoint").exists(),
    }
    if not all((
        report["original_hashes_match"],
        report["generation_configuration_unchanged"],
        report["eligible_pair_count"] == 8,
        report["remaining_attempt_allowance"] > 0,
        report["remaining_cost_allowance_usd"] > 0,
        report["api_key_available_but_not_recorded"],
        not report["held_out_checkpoint_exists"],
    )):
        raise RuntimeError("paid recovery pre-execution audit failed")
    destination = recovery / "PREEXECUTION_VALIDATION.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
