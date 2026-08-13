"""Aggregate completed paired pilots without rerunning paid inference.

This is deliberately limited to descriptive pilot reporting.  It converts
retrieval metrics to NaN on examples with no gold relevant evidence because
precision, recall, nDCG and MRR are undefined for those examples.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from sar_rag.experiments import classification_summary, paired_statistics, save_graphs


RETRIEVAL_METRICS = (
    "retrieval_precision_at_5", "retrieval_recall_at_5", "retrieval_ndcg_at_5",
    "retrieval_precision_at_10", "retrieval_recall_at_10", "retrieval_ndcg_at_10",
    "retrieval_precision_at_20", "retrieval_recall_at_20", "retrieval_ndcg_at_20",
    "retrieval_mrr", "evidence_set_precision", "evidence_set_recall",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records = []
    sources = []
    for source in args.inputs:
        payload = json.loads(source.read_text(encoding="utf-8"))
        sources.append(str(source))
        for record in payload["records"]:
            if record.get("gold_answerability") == "INSUFFICIENT":
                for method in ("traditional", "proposed"):
                    for metric in RETRIEVAL_METRICS:
                        record[method][metric] = float("nan")
            records.append(record)
    statistics = paired_statistics(records, 10_000, 20260803)
    summary = {
        "status": "pilot_not_for_claims",
        "examples": len(records),
        "source_result_files": sources,
        "answerability": {
            method: classification_summary(records, method)
            for method in ("traditional", "proposed")
        },
        "paired_statistics": statistics,
        "interpretation": (
            "Descriptive feasibility pilot only. Sample size is too small for "
            "inferential or superiority claims; confidence intervals and randomization "
            "p-values are not decision-valid at n=2."
        ),
        "metric_corrections": (
            "Retrieval metrics on examples with zero gold relevant evidence are NaN, "
            "not perfect scores."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    fields = ["example_id", "dataset", "method"] + sorted(
        set(records[0]["traditional"]) | set(records[0]["proposed"])
    )
    with (args.output_dir / "paired_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            for method in ("traditional", "proposed"):
                writer.writerow({
                    "example_id": record["example_id"],
                    "dataset": record.get("dataset"),
                    "method": method,
                    **record[method],
                })
    save_graphs(statistics, args.output_dir / "comparison.png")
    print(json.dumps({"output": str(args.output_dir), "examples": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
