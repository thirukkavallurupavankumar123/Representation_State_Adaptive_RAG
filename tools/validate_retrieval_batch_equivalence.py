"""Compare fixed-batch retrieval preparation with saved single-document records."""

from __future__ import annotations

import json
from pathlib import Path

import prepare_final_heldout_requests_v1_1 as prep


def main() -> int:
    report_path = prep.OUT / "RETRIEVAL_BATCH_EQUIVALENCE_VALIDATION.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite equivalence report: {report_path}")
    saved_rows = [json.loads(line) for line in prep.OFFLINE_CHECKPOINT.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = saved_rows[:12]
    questions = prep.load_jsonl(prep.QUESTIONS)
    questions_by_document: dict[str, list[dict]] = {}
    for question in questions:
        questions_by_document.setdefault(question["document_key"], []).append(question)
    documents = {row["document_key"]: row["documents"] for row in prep.load_jsonl(prep.DOCUMENTS)}
    tasks = [
        (row["document_key"], documents[row["document_key"]], sorted(questions_by_document[row["document_key"]], key=lambda value: value["question_id"]))
        for row in selected
    ]
    prep.initialize_worker()
    actual = prep.prepare_batch(tasks)
    expected = {
        row["document_key"]: [(item["request"], item["audit"]) for item in row["items"]]
        for row in selected
    }
    mismatches = [key for key, items in actual if items != expected[key]]
    result = {
        "status": "PASS" if not mismatches else "FAIL",
        "groups_checked": len(selected),
        "records_checked": sum(len(items) for _, items in actual),
        "exact_record_equality": not mismatches,
        "mismatched_document_keys": mismatches,
        "gold_labels_accessed": False,
        "api_calls": 0,
    }
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
