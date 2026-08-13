from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

from .dataset_adapters import iter_musique_full, iter_qasper


def normalize_question(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def profile(rows) -> dict:
    questions, example_ids, source_ids, document_ids = {}, set(), set(), set()
    duplicate_ids = []
    count = 0
    for row in rows:
        count += 1
        example_id = row["example_id"]
        if example_id in example_ids:
            duplicate_ids.append(example_id)
        example_ids.add(example_id)
        normalized = normalize_question(row["query"])
        questions.setdefault(sha256(normalized.encode()).hexdigest(), []).append(example_id)
        metadata = row.get("metadata", {})
        source_id = metadata.get("source_id") or metadata.get("paper_id")
        if source_id:
            source_ids.add(str(source_id))
        document_ids.update(document["document_id"] for document in row["documents"])
    return {
        "examples": count, "example_ids": example_ids, "questions": questions,
        "source_ids": source_ids, "document_ids": document_ids,
        "duplicate_example_ids": duplicate_ids,
    }


def overlap(left: dict, right: dict) -> dict:
    question_hashes = set(left["questions"]) & set(right["questions"])
    return {
        "exact_normalized_question_overlap": len(question_hashes),
        "overlap_question_examples": [
            {"left": left["questions"][key], "right": right["questions"][key]}
            for key in sorted(question_hashes)
        ][:100],
        "source_id_overlap": len(left["source_ids"] & right["source_ids"]),
        "document_id_overlap": len(left["document_ids"] & right["document_ids"]),
    }


def public(profile_value: dict) -> dict:
    return {
        "examples": profile_value["examples"],
        "unique_example_ids": len(profile_value["example_ids"]),
        "duplicate_example_ids": profile_value["duplicate_example_ids"],
        "unique_question_hashes": len(profile_value["questions"]),
        "unique_source_ids": len(profile_value["source_ids"]),
        "unique_document_ids": len(profile_value["document_ids"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Exact split-overlap and identifier leakage audit")
    parser.add_argument("--qasper-train", type=Path, required=True)
    parser.add_argument("--qasper-test", type=Path, required=True)
    parser.add_argument("--musique-train", type=Path, required=True)
    parser.add_argument("--musique-dev", type=Path, required=True)
    parser.add_argument("--musique-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    profiles = {
        "qasper_train": profile(iter_qasper(args.qasper_train, "train")),
        "qasper_test": profile(iter_qasper(args.qasper_test, "test")),
        "musique_train": profile(iter_musique_full(args.musique_train, "train")),
        "musique_dev": profile(iter_musique_full(args.musique_dev, "validation")),
    }
    registry = json.loads(args.musique_registry.read_text(encoding="utf-8"))
    report = {
        "scope": "exact normalized-question, source-ID, document-ID, and duplicate-ID audit; semantic paraphrase leakage is not ruled out",
        "splits": {name: public(value) for name, value in profiles.items()},
        "within_dataset": {
            "qasper_train_vs_test": overlap(profiles["qasper_train"], profiles["qasper_test"]),
            "musique_train_vs_dev": overlap(profiles["musique_train"], profiles["musique_dev"]),
        },
        "cross_dataset": {
            "qasper_train_vs_musique_dev": overlap(profiles["qasper_train"], profiles["musique_dev"]),
            "musique_train_vs_qasper_test": overlap(profiles["musique_train"], profiles["qasper_test"]),
        },
        "musique_official_dev_test_singlehop_registry": {
            "path": str(args.musique_registry),
            "sha256": sha256(args.musique_registry.read_bytes()).hexdigest(),
            "root_type": type(registry).__name__,
            "entries": len(registry),
            "purpose": "official registry retained for MuSiQue dev/test seed-question leakage controls",
        },
        "interpretation": "Zero exact overlap does not prove absence of semantic, pretraining, or benchmark contamination.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "splits": {k: v["examples"] for k, v in profiles.items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
