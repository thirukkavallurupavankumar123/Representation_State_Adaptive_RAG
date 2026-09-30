"""Compare two smoke runs with identical requests and seeds (repeatability diagnostic)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
first, second = HERE / sys.argv[1], HERE / sys.argv[2]


def load(directory: Path) -> dict:
    return {json.loads(l)["logical_key"]: json.loads(l) for l in (directory / "PREDICTIONS.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}


a, b = load(first), load(second)
rows = []
for key in sorted(a):
    x, y = a[key], b.get(key)
    rows.append({
        "logical_key": key,
        "same_request_fingerprint": y is not None and x["request_fingerprint"] == y["request_fingerprint"],
        "same_seed": y is not None and x["question_seed"] == y["question_seed"],
        "identical_raw_output": y is not None and x["raw_output"] == y["raw_output"],
        "identical_output_token_ids": y is not None and x["output_token_ids_sha256"] == y["output_token_ids_sha256"],
        "validation": [x["validation_status"], y["validation_status"] if y else None],
    })
report = {
    "runs": [first.name, second.name],
    "pairs_compared": len(rows),
    "identical_raw_outputs": sum(r["identical_raw_output"] for r in rows),
    "identical_output_token_ids": sum(r["identical_output_token_ids"] for r in rows),
    "all_fingerprints_equal": all(r["same_request_fingerprint"] for r in rows),
    "note": "Same hardware, software, model load path and seeds. Fixed seeds improve repeatability but do not guarantee identical outputs across hardware or library versions.",
    "pairs": rows,
}
(HERE / f"REPEATABILITY_{first.name}_vs_{second.name}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps({k: v for k, v in report.items() if k != "pairs"}, indent=2))
