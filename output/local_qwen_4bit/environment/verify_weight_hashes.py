"""Compare freshly computed local Qwen3-8B SHA-256 values with the hashes recorded at download time."""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
recorded = {}
with open("/mnt/f/research/environment/qwen3_8b_weight_hashes.csv", encoding="utf-8-sig") as stream:
    for row in csv.DictReader(stream):
        recorded[row["Path"].replace("\\", "/").rsplit("/", 1)[-1]] = row["Hash"].lower()
local = {}
lines = (HERE / "qwen3_8b_local_sha256.txt").read_text(encoding="utf-8").splitlines()
for line in lines[1:-1]:
    digest, name = line.split(maxsplit=1)
    local[name.lstrip("*")] = digest
checks = {name: local.get(name) == digest for name, digest in recorded.items()}
report = {
    "hash_started_utc": lines[0], "hash_finished_utc": lines[-1],
    "recorded_source": "/mnt/f/research/environment/qwen3_8b_weight_hashes.csv",
    "weights_checked": len(checks), "all_weights_match": all(checks.values()) and len(checks) == 5,
    "per_file": checks, "local_sha256": local,
    "conclusion": "local copy verified; no re-download required" if all(checks.values()) else "MISMATCH: investigate before use",
}
(HERE / "qwen3_8b_weight_verification.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({k: report[k] for k in ("weights_checked", "all_weights_match", "conclusion", "hash_started_utc", "hash_finished_utc")}))
