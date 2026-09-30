"""Show per-pair timing decomposition for a run directory (read-only)."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

directory = Path(sys.argv[1])
timings = {}
for line in (directory / "REQUEST_TIMINGS.jsonl").read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    timings[row["logical_key"]] = row["latency_end_to_end_including_save_seconds"]
print("now", datetime.now(timezone.utc).strftime("%H:%M:%S"))
for line in (directory / "PREDICTIONS.jsonl").read_text(encoding="utf-8").splitlines()[-int(sys.argv[2]) if len(sys.argv) > 2 else 0:]:
    row = json.loads(line)
    t = row["timing"]
    print(row["started_utc"][11:19], row["finished_utc"][11:19], f"e2e={timings.get(row['logical_key'], 0):6.2f}",
          f"render={t['render_and_tokenize_seconds']:.2f} gen={t['generate_seconds']:.2f} detok={t['detokenize_seconds']:.3f} "
          f"val={t['parse_and_validate_seconds']:.3f} before_save={row['latency_end_to_end_before_save_seconds']:.2f}")
