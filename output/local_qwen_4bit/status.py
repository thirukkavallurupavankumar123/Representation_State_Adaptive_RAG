"""Print a compact progress view of a local-Qwen run directory (read-only)."""
import json
import sys
from pathlib import Path

directory = Path(sys.argv[1])
for line in (directory / "RUN_LOG.jsonl").read_text(encoding="utf-8").splitlines():
    event = json.loads(line)
    print(event["utc"][11:19], event["event"], {k: v for k, v in event.items() if k in {"load_seconds", "warmup_seconds", "stop_reason", "new_pairs", "completed_pairs"}})
path = directory / "PREDICTIONS.jsonl"
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []
for row in rows[-int(sys.argv[2]) if len(sys.argv) > 2 else 0:]:
    problem = row["schema_errors"] or row["semantic_error"] or row["parse_error"] or ""
    print(f"{row['logical_key'][-58:]:58} {row['validation_status']:16} in={row['input_tokens']:5} out={row['output_tokens']:5} "
          f"gen={row['timing']['generate_seconds']:6.1f}s pre={row['timing']['prefill_seconds'] or 0:5.2f}s {str(problem)[:120]}")
from collections import Counter
print("completed:", len(rows), dict(Counter(row["validation_status"] for row in rows)))
