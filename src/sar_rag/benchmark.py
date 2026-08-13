from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

from .cli import result_to_dict
from .config import SarRagConfig
from .evaluation import evaluate_result, macro_average
from .environment import load_project_env
from .models import Document
from .pipeline import SarRagSystem


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {"example_id", "query", "documents", "gold_answers", "gold_evidence_spans"}
            missing = required - row.keys()
            if missing:
                raise ValueError(f"line {line_number} missing fields: {sorted(missing)}")
            rows.append(row)
    return rows


def _documents(row: dict) -> list[Document]:
    return [Document(**value) for value in row["documents"]]


def main(argv: list[str] | None = None) -> int:
    load_project_env()
    parser = argparse.ArgumentParser(description="Run a normalized SAR-RAG JSONL benchmark.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)

    rows = _load_jsonl(args.input)
    if args.limit is not None:
        rows = rows[:args.limit]
    outputs, metric_rows = [], []
    systems: dict[str, SarRagSystem] = {}
    for row in rows:
        documents = _documents(row)
        corpus_key = json.dumps([asdict(document) for document in documents], sort_keys=True)
        if corpus_key not in systems:
            systems[corpus_key] = SarRagSystem.research(
                documents, SarRagConfig(groq_model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"))
            )
        result = systems[corpus_key].ask(row["query"])
        metrics = evaluate_result(
            result,
            tuple(row["gold_answers"]),
            tuple(row["gold_evidence_spans"]),
            row.get("gold_answerability"),
        )
        metric_rows.append(metrics)
        outputs.append({"example_id": row["example_id"], "metrics": metrics, "result": result_to_dict(result)})

    payload = {"aggregate": macro_average(metric_rows), "examples": outputs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
