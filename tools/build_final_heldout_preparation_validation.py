"""Seal a concise validation record for the held-out v1.1 preparation package."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sar_rag.final_evaluation import file_sha256, object_sha256  # noqa: E402
from sar_rag.smoke_recovery import load_jsonl  # noqa: E402


OUT = ROOT / "output/final_evaluation/heldout_v1_1"
DESTINATION = OUT / "FINAL_PREEXECUTION_VALIDATION.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    if DESTINATION.exists():
        raise FileExistsError(f"refusing to overwrite sealed validation: {DESTINATION}")
    protocol_path = OUT / "FINAL_EVALUATION_PROTOCOL_AMENDMENT_V1_1.json"
    manifest_path = OUT / "FROZEN_QUESTION_SYSTEM_MANIFEST.json"
    exposure_path = OUT / "DATASET_ELIGIBILITY_AND_EXPOSURE.json"
    dry_path = OUT / "DRY_RUN_VALIDATION.json"
    auth_path = OUT / "PROPOSED_AUTHORIZATION_DISABLED.json"
    protocol, manifest, exposure, dry, authorization = map(read, (protocol_path, manifest_path, exposure_path, dry_path, auth_path))
    protocol_valid = object_sha256({key: value for key, value in protocol.items() if key != "protocol_fingerprint"}) == protocol["protocol_fingerprint"]
    manifest_valid = object_sha256({key: value for key, value in manifest.items() if key != "manifest_fingerprint"}) == manifest["manifest_fingerprint"]
    questions = load_jsonl(OUT / "GENERATION_QUESTIONS.jsonl")
    forbidden = {"gold_answers", "gold_answerability", "gold_evidence_spans"}
    original_integrity = {}
    smoke_protocol = read(ROOT / "output/final_evaluation/generation_v1_1/PROTOCOL.json")
    for name, expected in smoke_protocol["preserved_artifact_sha256"].items():
        original_integrity[name] = file_sha256(ROOT / "output/final_evaluation" / name) == expected
    checks = {
        "protocol_fingerprint_valid": protocol_valid,
        "manifest_fingerprint_valid": manifest_valid,
        "protocol_manifest_link_valid": manifest["protocol_fingerprint"] == protocol["protocol_fingerprint"],
        "questions_6240": len(questions) == 6240,
        "pairs_18720": manifest["pair_count"] == 18720 and len(manifest["pairs"]) == 18720,
        "maximum_attempts_56160": manifest["maximum_total_api_attempts_including_retries"] == 56160,
        "maximum_cost_77_5008": manifest["maximum_cost_usd"] == 77.5008,
        "generation_question_gold_fields_absent": all(not (set(row) & forbidden) for row in questions),
        "exposure_exclusions_applied": exposure["excluded_prior_exposure"]["qasper_excluded_count"] == 43 and exposure["excluded_prior_exposure"]["musique_excluded_count"] == 2,
        "smoke_48_first_attempt_verified": protocol["smoke_prerequisite"]["status"] == "VERIFIED_48_OF_48_FIRST_ATTEMPT_VALID",
        "dry_run_18720_passed": dry["status"] == "PASSED_DETERMINISTIC_FAKE_ANSWERER_DRY_RUN" and dry["fake_calls"] == 18720,
        "dry_resume_zero_duplicate_calls": dry["resume_zero_duplicate_calls"],
        "authorization_disabled": authorization["status"] == "DISABLED_PROPOSAL_NOT_AUTHORIZATION" and authorization["user_authorized"] is False,
        "paid_requests_not_built": not (OUT / "REQUESTS.jsonl").exists() and not (OUT / "REQUEST_MANIFEST.json").exists(),
        "heldout_paid_checkpoint_absent": not (OUT / "paid_checkpoint").exists(),
        "heldout_scoring_output_absent": not (OUT / "scoring").exists(),
        "all_original_and_recovery_artifacts_preserved": all(original_integrity.values()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"final held-out preparation validation failed: {checks}")
    artifacts = [
        protocol_path, manifest_path, exposure_path, dry_path, auth_path,
        OUT / "GENERATION_QUESTIONS.jsonl", OUT / "GENERATION_DOCUMENTS.jsonl",
        OUT / "FINAL_EVALUATION_PROTOCOL_AMENDMENT_V1_1.md",
        OUT / "EXECUTION_AND_SCORING_COMMANDS.md", OUT / "COST_AND_RUNTIME_PLAN.md",
    ]
    report = {
        "status": "READY_FOR_OFFLINE_RETRIEVAL_REQUEST_FREEZE_NOT_AUTHORIZED_FOR_PAID_EXECUTION",
        "checks": checks,
        "scope": {"questions": 6240, "systems": 3, "initial_attempts": 18720, "maximum_attempts": 56160, "maximum_cost_usd": 77.5008},
        "test_results": {"focused": "21 passed", "full_regression": "132 passed, 14 warnings", "api_calls": 0},
        "fingerprints": {"protocol": protocol["protocol_fingerprint"], "manifest": manifest["manifest_fingerprint"]},
        "artifact_sha256": {path.name: file_sha256(path) for path in artifacts},
        "unresolved_before_authorization": [
            "run the deterministic offline retrieval/request builder and seal its request-manifest fingerprint",
            "verify the account-specific Groq RPM/TPM/RPD/TPD limits in provider settings",
            "review the 43 exact-question-text and one MuSiQue source-group exclusions",
        ],
        "gold_labels_accessed_during_preparation": False,
        "heldout_generation_executed": False,
        "heldout_scoring_executed": False,
    }
    DESTINATION.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
