"""Print invalid generations with their supplied evidence IDs (read-only; design-exposed smoke only)."""
import json
import sys
from pathlib import Path

directory = Path(sys.argv[1])
for line in (directory / "PREDICTIONS.jsonl").read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    if row["validation_status"] != "valid":
        print("=" * 100)
        print(row["logical_key"], row["validation_status"], row["finish_reason"])
        print("supplied ids:", row["supplied_evidence_ids"])
        print("errors:", row["schema_errors"], row["semantic_error"], row["parse_error"])
        print("RAW:", row["raw_output"])
summary = json.loads((directory / "SMOKE_SUMMARY.json").read_text(encoding="utf-8")) if (directory / "SMOKE_SUMMARY.json").exists() else None
if summary:
    print(json.dumps({k: v for k, v in summary.items() if k not in {"per_pair", "run_summary"}}, indent=1))
