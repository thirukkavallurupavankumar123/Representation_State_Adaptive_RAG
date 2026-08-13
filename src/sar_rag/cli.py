from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
import os
from pathlib import Path

from .config import SarRagConfig
from .environment import load_project_env
from .models import Document, EvidenceItem, PipelineResult
from .pipeline import SarRagSystem


def _document_from_path(path: Path) -> Document:
    text = path.read_text(encoding="utf-8")
    digest = sha256(text.encode("utf-8")).hexdigest()
    return Document(
        document_id=sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:24],
        version_id=digest[:24],
        title=path.stem,
        text=text,
    )


def _evidence_json(item: EvidenceItem, number: int) -> dict:
    return {
        "evidence_id": f"E{number}",
        "document_id": item.candidate.document_id,
        "version_id": item.candidate.version_id,
        "canonical_unit_ids": list(item.candidate.canonical_unit_ids),
        "representation": item.candidate.representation_id,
        "rank": item.candidate.rank,
        "retrieval_stage": item.candidate.retrieval_stage,
        "text": item.chunk.evidence_text,
        "obligations": sorted(item.obligations),
        "provenance": [list(value) for value in item.provenance],
    }


def result_to_dict(result: PipelineResult) -> dict:
    return {
        "query": asdict(result.query),
        "selected_representations": list(result.selected_representations),
        "representation_profiles": {
            key: {
                "obligations": sorted(value.obligations),
                "exclusive_obligations": sorted(value.exclusive_obligations),
                "rrf_contribution": value.rrf_contribution,
                "normalized_coverage_score": value.normalized_coverage_score,
                "marginal_gain": value.marginal_gain,
                "mean_probe_words": value.mean_probe_words,
                "selected": value.selected,
                "selection_reasons": value.selection_reasons,
            }
            for key, value in result.representation_profiles.items()
        },
        "final_k": result.final_k,
        "answerability": result.answerability,
        "unresolved_requirements": list(result.unresolved_requirements),
        "stop_reason": result.stop_reason,
        "answer": result.answer,
        "evidence": [_evidence_json(item, index) for index, item in enumerate(result.final_evidence, 1)],
        "trace": result.trace,
        "run_metadata": result.run_metadata,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run representation-state adaptive RAG.")
    parser.add_argument("--document", action="append", required=True, type=Path,
                        help="UTF-8 .txt or .md source; repeat for multiple documents")
    parser.add_argument("--query", required=True)
    parser.add_argument("--output", type=Path, help="Write the complete JSON result here")
    parser.add_argument("--embedding-revision", help="Optional exact Hugging Face model commit")
    parser.add_argument("--groq-model", default="openai/gpt-oss-120b")
    parser.add_argument("--probe-depth", type=int, default=3,
                        help="Diagnostic depth per representation; this is not final retrieval k")
    parser.add_argument("--candidate-budget", type=int, default=60)
    parser.add_argument("--context-token-budget", "--context-word-budget",
                        dest="context_token_budget", type=int, default=2000,
                        help="Final evidence tokenizer-token budget; old word-budget name is an alias")
    parser.add_argument("--similarity-threshold", type=float, default=0.70,
                        help="Per-representation empirical relevance percentile")
    parser.add_argument("--similarity-gap-threshold", type=float, default=0.75,
                        help="Score drop divided by representation score standard deviation")
    parser.add_argument("--novelty-threshold", type=float, default=0.10)
    parser.add_argument("--marginal-utility-threshold", type=float, default=0.05)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_project_env()
    args = build_parser().parse_args(argv)
    missing = [str(path) for path in args.document if not path.is_file()]
    if missing:
        raise SystemExit("Missing document(s): " + ", ".join(missing))
    config = replace(
        SarRagConfig(),
        groq_model=os.environ.get("GROQ_MODEL", args.groq_model),
        probe_depth=args.probe_depth,
        max_candidates_examined=args.candidate_budget,
        max_final_context_tokens=args.context_token_budget,
        similarity_threshold=args.similarity_threshold,
        similarity_gap_threshold=args.similarity_gap_threshold,
        novelty_threshold=args.novelty_threshold,
        marginal_utility_threshold=args.marginal_utility_threshold,
    )
    documents = [_document_from_path(path) for path in args.document]
    system = SarRagSystem.research(documents, config, embedding_revision=args.embedding_revision)
    payload = result_to_dict(system.ask(args.query))
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0
