from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable, Iterator


class DatasetFormatError(ValueError):
    pass


def _stable_version(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()[:24]


def _clean_heading(value: str) -> str:
    return " ".join(str(value).replace("#", " ").split()) or "Untitled"


def _unique(values: Iterable[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        normalized = " ".join(str(value).split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def iter_qasper(path: Path, split: str) -> Iterator[dict]:
    """Convert the official QASPER v0.3 JSON without changing source paragraphs."""
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise DatasetFormatError("QASPER root must be an object keyed by paper ID")
    for paper_id, paper in data.items():
        required = {"title", "abstract", "full_text", "qas"}
        if not isinstance(paper, dict) or not required <= paper.keys():
            raise DatasetFormatError(f"QASPER paper {paper_id} has an unexpected schema")
        parts = [f"# {_clean_heading(paper['title'])}"]
        abstract = str(paper["abstract"]).strip()
        if abstract:
            parts.extend(["## Abstract", abstract])
        for section in paper["full_text"]:
            if not isinstance(section, dict) or not {"section_name", "paragraphs"} <= section.keys():
                raise DatasetFormatError(f"QASPER paper {paper_id} has an invalid section")
            parts.append(f"## {_clean_heading(section['section_name'])}")
            parts.extend(str(paragraph).strip() for paragraph in section["paragraphs"] if str(paragraph).strip())
        document_text = "\n\n".join(parts)
        document = {
            "document_id": f"qasper:{paper_id}",
            "version_id": _stable_version(document_text),
            "title": str(paper["title"]),
            "text": document_text,
            "effective_date": None,
            "authority_level": "published_research_paper",
            "supersedes_version_id": None,
        }
        for qa in paper["qas"]:
            if not isinstance(qa, dict) or not {"question", "question_id", "answers"} <= qa.keys():
                raise DatasetFormatError(f"QASPER paper {paper_id} has an invalid QA record")
            answers, evidence, unanswerable_flags = [], [], []
            for annotation in qa["answers"]:
                answer = annotation.get("answer", {})
                if not isinstance(answer, dict):
                    raise DatasetFormatError(f"QASPER question {qa['question_id']} has an invalid answer")
                unanswerable = bool(answer.get("unanswerable", False))
                unanswerable_flags.append(unanswerable)
                if not unanswerable:
                    if answer.get("yes_no") is not None:
                        answers.append("yes" if answer["yes_no"] else "no")
                    elif str(answer.get("free_form_answer", "")).strip():
                        answers.append(str(answer["free_form_answer"]))
                    else:
                        answers.extend(str(value) for value in answer.get("extractive_spans", []))
                evidence.extend(str(value) for value in answer.get("evidence", []))
            unique_answers = _unique(answers)
            if unique_answers:
                gold_answerability = "ANSWERABLE"
            elif unanswerable_flags and all(unanswerable_flags):
                gold_answerability = "INSUFFICIENT"
            else:
                gold_answerability = None
            row = {
                "example_id": f"qasper:{split}:{qa['question_id']}",
                "dataset": "qasper",
                "split": split,
                "query": str(qa["question"]),
                "documents": [document],
                "gold_answers": unique_answers,
                "gold_evidence_spans": _unique(evidence),
                "metadata": {"paper_id": str(paper_id)},
            }
            if gold_answerability is not None:
                row["gold_answerability"] = gold_answerability
            yield row


def iter_musique_full(path: Path, split: str) -> Iterator[dict]:
    """Convert the official MuSiQue-Full v1.0 JSONL, preserving each paragraph."""
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            required = {"id", "paragraphs", "question"}
            if not isinstance(item, dict) or not required <= item.keys():
                raise DatasetFormatError(f"MuSiQue line {line_number} has an unexpected schema")
            documents, evidence = [], []
            for paragraph in item["paragraphs"]:
                if not isinstance(paragraph, dict) or not {"idx", "title", "paragraph_text"} <= paragraph.keys():
                    raise DatasetFormatError(f"MuSiQue line {line_number} has an invalid paragraph")
                text = f"# {_clean_heading(paragraph['title'])}\n\n{str(paragraph['paragraph_text']).strip()}"
                documents.append({
                    "document_id": f"musique:{item['id']}:p{paragraph['idx']}",
                    "version_id": _stable_version(text),
                    "title": str(paragraph["title"]),
                    "text": text,
                    "effective_date": None,
                    "authority_level": "provided_context_paragraph",
                    "supersedes_version_id": None,
                })
                if paragraph.get("is_supporting", False):
                    evidence.append(str(paragraph["paragraph_text"]))
            labels_available = {"answer", "answer_aliases", "answerable"} <= item.keys()
            answerable = bool(item["answerable"]) if labels_available else None
            variant = (
                "answerable" if answerable else "insufficient"
            ) if labels_available else "hidden"
            gold_answers = (
                _unique([str(item["answer"]), *(str(x) for x in item["answer_aliases"])])
                if answerable else []
            )
            row = {
                # MuSiQue-Full may pair answerable and insufficient variants under
                # the same official source ID. Include the variant so checkpoints,
                # sampling and paired statistics never collapse distinct examples.
                "example_id": f"musique_full:{split}:{item['id']}:{variant}",
                "dataset": "musique_full",
                "split": split,
                "query": str(item["question"]),
                "documents": documents,
                "gold_answers": gold_answers,
                "gold_evidence_spans": _unique(evidence),
                "metadata": {
                    "source_id": str(item["id"]),
                    "variant": variant,
                    "official_labels_hidden": not labels_available,
                    "latent_answer_for_unanswerable_group": (
                        str(item["answer"]) if labels_available and not answerable else None
                    ),
                },
            }
            if labels_available:
                row["gold_answerability"] = "ANSWERABLE" if answerable else "INSUFFICIENT"
            yield row


def _default_input(dataset: str, split: str) -> Path:
    root = Path(__file__).resolve().parents[2] / "datasets" / "raw"
    if dataset == "qasper":
        qasper_split = "dev" if split == "validation" else split
        return root / "qasper" / "data" / f"qasper-{qasper_split}-v0.3.json"
    musique_split = "dev" if split == "validation" else split
    return root / "musique" / "data" / f"musique_full_v1.0_{musique_split}.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Normalize an official research dataset for SAR-RAG.")
    parser.add_argument("--dataset", choices=["qasper", "musique_full"], required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], required=True)
    parser.add_argument("--input", type=Path, help="Override the verified default raw-data path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Deterministic first-N smoke/calibration subset")
    args = parser.parse_args(argv)
    source = args.input or _default_input(args.dataset, args.split)
    if not source.is_file():
        raise SystemExit(f"Dataset source not found: {source}")
    iterator = iter_qasper(source, args.split) if args.dataset == "qasper" else iter_musique_full(source, args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as stream:
        for row in iterator:
            if args.limit is not None and count >= args.limit:
                break
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    print(json.dumps({"dataset": args.dataset, "split": args.split, "examples": count,
                      "input": str(source), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
