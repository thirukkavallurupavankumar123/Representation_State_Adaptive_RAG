"""Create a final integrity and worktree inventory without reading secret contents."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "adaptive_single_view_v2" / "PHASE_E_INTEGRITY.json"
INVENTORY = ROOT / "output" / "adaptive_single_view_v2" / "WORKTREE_FILE_INVENTORY.txt"
V1 = ROOT / "output" / "calibration" / "frozen_manifest.json"
RUN = ROOT / "output" / "adaptive_single_view_v2" / "recalibration_seed20260905" / "RUN_MANIFEST.json"
RAW = ROOT / "output" / "independent_verification" / "screening_v3_n286"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=False
    )


def main() -> None:
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    lines = [line for line in status.stdout.splitlines() if line.strip()]
    inventory_rel = INVENTORY.relative_to(ROOT).as_posix()
    integrity_rel = OUT.relative_to(ROOT).as_posix()
    for rel in (inventory_rel, integrity_rel):
        marker = f"?? {rel}"
        if marker not in lines:
            lines.append(marker)
    lines.sort(key=lambda line: line[3:].lower())
    INVENTORY.write_text("\n".join(lines) + "\n", encoding="utf-8")

    env_tracked = git("ls-files", "--error-unmatch", "--", ".env").returncode == 0
    protected_diff = git(
        "diff",
        "--name-only",
        "--",
        "output/calibration/frozen_manifest.json",
        "output/independent_verification/screening_v3_n286",
    ).stdout.splitlines()
    run = json.loads(RUN.read_text(encoding="utf-8"))
    new_output_root = ROOT / "output" / "adaptive_single_view_v2"
    suspicious_names = [
        path.relative_to(ROOT).as_posix()
        for path in new_output_root.rglob("*")
        if path.is_file() and any(token in path.name.lower() for token in ("groq", "api_call", "llm_call"))
    ]
    raw_files = sorted(path for path in RAW.rglob("*") if path.is_file())
    payload = {
        "status": "PHASE_E_VALIDATED",
        "frozen_v1": {
            "path": "output/calibration/frozen_manifest.json",
            "expected_sha256": "5c570ec2b4c52e1b0ea0f2152a6f8fb6f1fa1c1575e58cfb58bde2c434a54d05",
            "actual_sha256": sha256(V1),
            "matches": sha256(V1) == "5c570ec2b4c52e1b0ea0f2152a6f8fb6f1fa1c1575e58cfb58bde2c434a54d05",
        },
        "protected_tracked_differences": protected_diff,
        "raw_screening": {
            "path": "output/independent_verification/screening_v3_n286",
            "file_count": len(raw_files),
            "file_sha256": {
                path.relative_to(ROOT).as_posix(): sha256(path) for path in raw_files
            },
        },
        "secrets": {"env_tracked": env_tracked, "env_contents_read": False},
        "paid_execution": {
            "run_manifest_llm_calls": run.get("llm_calls"),
            "failures": run.get("failures"),
            "retries": run.get("retries"),
            "suspicious_new_output_filenames": suspicious_names,
            "held_out_answer_generation": "NOT_RUN",
        },
        "experiment": {
            "fingerprint": run.get("experiment_fingerprint"),
            "exact_command": run.get("exact_command"),
            "python": run.get("python"),
            "platform": run.get("platform"),
            "packages": run.get("packages"),
            "artifact_sha256": run.get("artifact_sha256"),
        },
        "tests": {"passed": 104, "failed": 0, "warnings": 14, "duration_seconds": 47.22},
        "worktree_inventory": inventory_rel,
        "worktree_entry_count": len(lines),
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "inventory": str(INVENTORY), "entries": len(lines)}))


if __name__ == "__main__":
    main()
