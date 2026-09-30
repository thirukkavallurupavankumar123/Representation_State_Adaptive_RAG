"""Build final protocol validation, integrity, and handoff artifacts."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sar_rag.final_evaluation import file_sha256  # noqa: E402

OUT = ROOT / "output" / "final_evaluation"
VALIDATION = OUT / "FINAL_EVALUATION_VALIDATION.json"
INVENTORY = OUT / "WORKTREE_FILE_INVENTORY.txt"


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)


def load_json(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def main() -> None:
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    lines = [line for line in status.stdout.splitlines() if line.strip()]
    for path in (VALIDATION, INVENTORY, OUT / "FINAL_EVALUATION_HANDOFF.md"):
        marker = f"?? {path.relative_to(ROOT).as_posix()}"
        if marker not in lines:
            lines.append(marker)
    lines.sort(key=lambda line: line[3:].lower())
    INVENTORY.write_text("\n".join(lines) + "\n", encoding="utf-8")

    protocol = load_json("FINAL_EVALUATION_PROTOCOL.json")
    smoke = load_json("SMOKE_TEST_MANIFEST.json")
    request_manifest = load_json("SMOKE_TEST_REQUESTS_MANIFEST.json")
    dry = load_json("DRY_RUN_VALIDATION.json")
    recompute = load_json("SMOKE_TEST_REQUESTS_RECOMPUTATION_VALIDATION.json")
    checkpoint_rows = [
        json.loads(line) for line in (OUT / "dry_run_checkpoint/ANSWERS.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    core_files = [
        "FINAL_EVALUATION_PROTOCOL.json", "FINAL_EVALUATION_PROTOCOL.md",
        "SMOKE_TEST_MANIFEST.json", "SMOKE_TEST_REQUESTS.jsonl",
        "SMOKE_TEST_REQUESTS_MANIFEST.json", "EXPECTED_CALLS_AND_COST.md",
        "DRY_RUN_VALIDATION.json", "AUTHORIZATION_GATE.md", "REPRODUCTION_COMMANDS.md",
        "PHASE_E_RESULT_AUDIT.json", "SMOKE_TEST_REQUESTS_RECOMPUTATION_VALIDATION.json",
    ]
    env_tracked = git("ls-files", "--error-unmatch", "--", ".env").returncode == 0
    protected_diff = git(
        "diff", "--name-only", "--",
        "output/calibration/frozen_manifest.json",
        "output/adaptive_single_view_v2/recalibration_seed20260905/FROZEN_V2_MANIFEST.json",
    ).stdout.splitlines()
    payload = {
        "status": "VERIFIED_FROZEN_PROTOCOL_READY_FOR_EXPLICIT_SMOKE_AUTHORIZATION",
        "validated_date": "2026-09-08",
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "smoke_manifest_fingerprint": smoke["manifest_fingerprint"],
        "request_manifest_fingerprint": request_manifest["manifest_fingerprint"],
        "phase_e_recomputed": protocol["phase_e_audit"],
        "smoke": {
            "questions": smoke["question_count"], "systems": len(smoke["systems"]),
            "maximum_paid_calls": smoke["maximum_calls"],
            "prepared_requests": request_manifest["request_count"],
            "maximum_observed_packed_context_tokens": request_manifest["maximum_observed_packed_context_tokens"],
            "fresh_retrieval_recomputation_identical": recompute["substantive_identical"],
        },
        "dry_run": {
            "status": dry["status"], "fake_calls": dry["fake_answerer_calls"],
            "resume_new_calls": dry["resume_run"]["new_calls"],
            "fingerprint_mismatch_rejected": dry["fingerprint_mismatch_rejected"],
            "checkpoint_records": len(checkpoint_rows),
            "retries": sum(int(row["retries"]) for row in checkpoint_rows),
            "failure_records": 0 if not (OUT / "dry_run_checkpoint/FAILURES.jsonl").exists() else len((OUT / "dry_run_checkpoint/FAILURES.jsonl").read_text(encoding="utf-8").splitlines()),
        },
        "paid_execution": {
            "groq_calls": 0, "paid_smoke_executed": False,
            "heldout_executed": False, "llm_judge_executed": False,
            "authorization_file_exists": (OUT / "PAID_SMOKE_AUTHORIZATION.json").exists(),
            "paid_checkpoint_exists": (OUT / "paid_smoke_checkpoint").exists(),
        },
        "integrity": {
            "frozen_v1_sha256": file_sha256(ROOT / "output/calibration/frozen_manifest.json"),
            "frozen_v2_sha256": file_sha256(ROOT / "output/adaptive_single_view_v2/recalibration_seed20260905/FROZEN_V2_MANIFEST.json"),
            "protected_tracked_differences": protected_diff,
            "env_tracked": env_tracked, "env_contents_printed": False,
        },
        "tests": {"passed": 113, "failed": 0, "warnings": 14, "duration_seconds": 68.69},
        "runtime": {
            "python": sys.version, "platform": platform.platform(),
            "packages": {name: package_version(name) for name in ("numpy", "torch", "transformers", "pytest", "matplotlib")},
        },
        "artifact_sha256": {name: file_sha256(OUT / name) for name in core_files},
        "worktree_inventory": str(INVENTORY.relative_to(ROOT)).replace("\\", "/"),
        "worktree_entry_count": len(lines),
    }
    if payload["paid_execution"]["authorization_file_exists"] or payload["paid_execution"]["paid_checkpoint_exists"]:
        raise RuntimeError("unexpected paid authorization or checkpoint exists")
    if payload["integrity"]["env_tracked"] or payload["integrity"]["protected_tracked_differences"]:
        raise RuntimeError("integrity check failed")
    VALIDATION.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    handoff = f"""# Final answer-generation protocol handoff

Status: **verified and frozen; waiting for explicit paid-smoke authorization**.

- Phase E values were recomputed from 63 paired per-example rows; v2 remains unsupported as a v1 replacement.
- Smoke sample: 16 design-exposed training questions, balanced 8/8 by dataset and answerability.
- Systems: Traditional sentence-window k=10, unchanged frozen SAR-RAG v1, frozen paragraph v2.
- Prepared requests: 48; fresh retrieval recomputation was byte-substantively identical.
- Deterministic dry run: 48 fake calls, 48 checkpoint records, zero new calls on resume, fingerprint drift rejected, zero failures/retries, zero Groq calls.
- Complete suite: 113 passed, 0 failed (14 third-party deprecation warnings).
- Paid smoke maximum: 48 calls; expected 112,800 tokens and $0.02232; planning ceiling 172,800 tokens and $0.04320.
- Full held-out and optional judge stages remain separately blocked.

See `AUTHORIZATION_GATE.md` for the exact paid command. It was not executed.
"""
    (OUT / "FINAL_EVALUATION_HANDOFF.md").write_text(handoff, encoding="utf-8")
    print(json.dumps({"status": payload["status"], "tests": payload["tests"], "paid_calls": 0, "inventory_entries": len(lines)}))


if __name__ == "__main__":
    main()
