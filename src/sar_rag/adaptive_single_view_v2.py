"""Adaptive Single-View Retrieval V2.

The retrieval path is mathematical and deterministic.  It never imports or
calls Groq.  Answer generation is deliberately outside this module.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Iterable, Mapping, Sequence

from .benchmark import _documents, _load_jsonl
from .calibration import _covered_spans
from .corpus import CorpusArtifacts, build_corpus
from .encoding import Encoder, SentenceTransformerEncoder
from .models import Candidate, Chunk, Document, EvidenceItem, Judgment

REPRESENTATIONS = ("semantic", "paragraph", "hierarchical_v2", "sentence_window")
TOKEN_OPERATING_POINTS = (800, 1200, 1600)
TOKEN_SAFETY_MAXIMUM = 2000


def canonical_sha256(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class AdaptiveSingleViewV2Config:
    representation: str
    max_context_tokens: int
    relevance_percentile_threshold: float = 0.55
    empty_batch_patience: int = 2
    probe_depth: int = 3
    adaptive_batch_size: int = 3
    max_candidates_examined: int = 60
    max_retrieval_rounds: int = 8
    max_final_chunks: int = 20
    adaptive_retrieval_enabled: bool = True
    deduplicate_after_retrieval: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    local_embedding_path: str = "models/all-MiniLM-L6-v2"
    seed: int = 20260905

    def __post_init__(self) -> None:
        if self.representation not in REPRESENTATIONS:
            raise ValueError(f"representation must be exactly one of {REPRESENTATIONS}")
        if self.max_context_tokens not in TOKEN_OPERATING_POINTS and self.max_context_tokens != TOKEN_SAFETY_MAXIMUM:
            raise ValueError(f"max_context_tokens must be a declared operating point or {TOKEN_SAFETY_MAXIMUM}")
        if not 0 <= self.relevance_percentile_threshold <= 1:
            raise ValueError("relevance_percentile_threshold must be in [0, 1]")
        for name in ("empty_batch_patience", "probe_depth", "adaptive_batch_size",
                     "max_candidates_examined", "max_retrieval_rounds", "max_final_chunks"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.max_candidates_examined < self.probe_depth:
            raise ValueError("candidate cap must cover the single-view probe")

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(asdict(self))


@dataclass
class V2RetrievalResult:
    query: str
    representation: str
    final_evidence: list[EvidenceItem]
    provisional_evidence: list[EvidenceItem]
    candidates_examined: int
    retrieval_rounds: int
    pre_dedup_k: int
    post_dedup_k: int
    final_k: int
    context_tokens: int
    stop_reason: str
    thresholds: dict[str, float | int | bool]
    configuration_fingerprint: str
    embedding_fingerprint: str
    trace: list[dict]


class AdaptiveSingleViewV2System:
    """A deterministic, one-representation adaptive retriever."""

    def __init__(self, artifacts: CorpusArtifacts, encoder: Encoder,
                 config: AdaptiveSingleViewV2Config, embedding_fingerprint: str):
        if set(artifacts.indexes) != {config.representation}:
            raise ValueError("v2 corpus must contain exactly the configured single representation")
        self.artifacts = artifacts
        self.encoder = encoder
        self.config = config
        self.embedding_fingerprint = embedding_fingerprint

    @classmethod
    def research(cls, documents: Iterable[Document], config: AdaptiveSingleViewV2Config) -> "AdaptiveSingleViewV2System":
        local = Path(config.local_embedding_path)
        if not local.is_absolute():
            local = Path(__file__).resolve().parents[2] / local
        source = str(local.resolve()) if local.is_dir() else config.embedding_model
        encoder = SentenceTransformerEncoder(source, revision=config.embedding_revision)
        weight = local / "model.safetensors"
        embedding = {
            "model": config.embedding_model, "revision": config.embedding_revision,
            "dimension": encoder.dimension, "source": source,
            "weight_sha256": file_sha256(weight) if weight.is_file() else None,
        }
        artifacts = build_corpus(documents, encoder, (config.representation,))
        return cls(artifacts, encoder, config, canonical_sha256(embedding))

    def _item(self, candidate: Candidate) -> EvidenceItem:
        chunk = self.artifacts.chunks_by_id[candidate.chunk_id]
        return EvidenceItem(candidate, chunk, Judgment(candidate.candidate_id, ()),
                            {f"UNIT:{unit_id}" for unit_id in candidate.canonical_unit_ids},
                            [(candidate.representation_id, candidate.rank)])

    def _deduplicate(self, scored: list[tuple[EvidenceItem, float, int]]) -> list[tuple[EvidenceItem, float, int]]:
        """Exact/containment dedup after retrieval, independent of stopping."""
        retained: list[tuple[EvidenceItem, float, int]] = []
        for item, relevance, tokens in scored:
            units = set(item.candidate.canonical_unit_ids)
            duplicate = None
            for position, (other, other_relevance, other_tokens) in enumerate(retained):
                if (item.candidate.document_id, item.candidate.version_id) != (
                    other.candidate.document_id, other.candidate.version_id
                ):
                    continue
                other_units = set(other.candidate.canonical_unit_ids)
                if units == other_units or units < other_units or other_units < units:
                    duplicate = position
                    current_key = (relevance, -tokens, -item.candidate.rank, item.candidate.candidate_id)
                    other_key = (other_relevance, -other_tokens, -other.candidate.rank,
                                 other.candidate.candidate_id)
                    if current_key > other_key:
                        item.provenance.extend(other.provenance)
                        retained[position] = (item, relevance, tokens)
                    else:
                        other.provenance.extend(item.provenance)
                    break
            if duplicate is None:
                retained.append((item, relevance, tokens))
        return retained

    def retrieve(self, query: str) -> V2RetrievalResult:
        config = self.config
        index = self.artifacts.indexes[config.representation]
        trace: list[dict] = [{
            "event": "v2_retrieval_started", "representation": config.representation,
            "configuration_fingerprint": config.fingerprint,
            "embedding_fingerprint": self.embedding_fingerprint,
            "llm_calls": 0,
        }]
        accepted: list[tuple[EvidenceItem, float, int]] = []
        seen_chunks: set[str] = set()
        accepted_tokens = 0
        candidates_examined = 0
        retrieval_rounds = 0
        stop_reason = "index_exhausted"

        def examine(candidate: Candidate) -> str:
            nonlocal accepted_tokens, candidates_examined
            candidates_examined += 1
            chunk = self.artifacts.chunks_by_id[candidate.chunk_id]
            tokens = self.encoder.token_count(chunk.evidence_text)
            relevance = index.relevance_percentile(query, candidate.raw_score)
            if candidate.chunk_id in seen_chunks:
                outcome = "duplicate_candidate"
            elif accepted_tokens + tokens > config.max_context_tokens:
                outcome = "context_token_safety_cap"
            elif len(accepted) >= config.max_final_chunks:
                outcome = "final_chunk_safety_cap"
            elif relevance < config.relevance_percentile_threshold:
                seen_chunks.add(candidate.chunk_id)
                outcome = "below_relevance_threshold"
            else:
                seen_chunks.add(candidate.chunk_id)
                accepted.append((self._item(candidate), relevance, tokens))
                accepted_tokens += tokens
                outcome = "accepted"
            trace.append({
                "event": "v2_candidate_scored_without_llm", "stage": candidate.retrieval_stage,
                "rank": candidate.rank, "chunk_id": candidate.chunk_id, "raw_score": candidate.raw_score,
                "normalized_relevance": relevance, "tokens": tokens, "outcome": outcome,
                "accepted_tokens_before_dedup": accepted_tokens,
            })
            return outcome

        probe = index.search(query, 1, min(config.probe_depth, config.max_candidates_examined), "probe")
        trace.append({"event": "v2_probe_retrieved", "candidate_count": len(probe),
                      "probe_depth": config.probe_depth})
        terminal = None
        for candidate in probe:
            outcome = examine(candidate)
            if outcome in {"context_token_safety_cap", "final_chunk_safety_cap"}:
                terminal = outcome
                break
        next_rank = len(probe) + 1
        empty_batches = 0

        if terminal:
            stop_reason = terminal
        elif not config.adaptive_retrieval_enabled:
            stop_reason = "probe_only_comparator"
        else:
            while True:
                if candidates_examined >= config.max_candidates_examined:
                    stop_reason = "candidate_safety_cap"
                    break
                if retrieval_rounds >= config.max_retrieval_rounds:
                    stop_reason = "retrieval_round_safety_cap"
                    break
                if len(accepted) >= config.max_final_chunks:
                    stop_reason = "final_chunk_safety_cap"
                    break
                retrieval_rounds += 1
                limit = min(config.adaptive_batch_size,
                            config.max_candidates_examined - candidates_examined)
                batch = index.search(query, next_rank, limit, "adaptive")
                next_rank += len(batch)
                if not batch:
                    stop_reason = "index_exhausted"
                    trace.append({"event": "v2_batch_evaluated_without_llm", "round": retrieval_rounds,
                                  "batch_size": 0, "accepted": 0,
                                  "consecutive_empty_batches": empty_batches})
                    break
                added = 0
                terminal = None
                for candidate in batch:
                    outcome = examine(candidate)
                    added += int(outcome == "accepted")
                    if outcome in {"context_token_safety_cap", "final_chunk_safety_cap"}:
                        terminal = outcome
                        break
                empty_batches = 0 if added else empty_batches + 1
                trace.append({"event": "v2_batch_evaluated_without_llm", "round": retrieval_rounds,
                              "batch_size": len(batch), "accepted": added,
                              "consecutive_empty_batches": empty_batches})
                if terminal:
                    stop_reason = terminal
                    break
                if empty_batches >= config.empty_batch_patience:
                    stop_reason = "patience_no_gain"
                    break

        pre_dedup = list(accepted)
        retained = self._deduplicate(pre_dedup) if config.deduplicate_after_retrieval else list(pre_dedup)
        retained.sort(key=lambda value: (-value[1], value[0].candidate.rank,
                                         value[0].candidate.candidate_id))
        final_evidence = [item for item, _, _ in retained]
        final_tokens = sum(tokens for _, _, tokens in retained)
        trace.append({
            "event": "v2_post_retrieval_deduplication_completed",
            "pre_dedup_k": len(pre_dedup), "post_dedup_k": len(retained),
            "deduplication_enabled": config.deduplicate_after_retrieval,
        })
        trace.append({
            "event": "v2_retrieval_completed_without_llm", "representation": config.representation,
            "candidates_examined": candidates_examined, "retrieval_rounds": retrieval_rounds,
            "pre_dedup_k": len(pre_dedup), "post_dedup_k": len(retained),
            "final_k": len(final_evidence), "context_tokens": final_tokens,
            "stop_reason": stop_reason, "llm_calls": 0,
        })
        return V2RetrievalResult(
            query=query, representation=config.representation, final_evidence=final_evidence,
            provisional_evidence=[item for item, _, _ in pre_dedup],
            candidates_examined=candidates_examined, retrieval_rounds=retrieval_rounds,
            pre_dedup_k=len(pre_dedup), post_dedup_k=len(retained), final_k=len(final_evidence),
            context_tokens=final_tokens, stop_reason=stop_reason,
            thresholds={
                "relevance_percentile_threshold": config.relevance_percentile_threshold,
                "empty_batch_patience": config.empty_batch_patience,
                "probe_depth": config.probe_depth, "adaptive_batch_size": config.adaptive_batch_size,
                "max_context_tokens": config.max_context_tokens,
                "max_candidates_examined": config.max_candidates_examined,
                "max_retrieval_rounds": config.max_retrieval_rounds,
                "max_final_chunks": config.max_final_chunks,
            }, configuration_fingerprint=config.fingerprint,
            embedding_fingerprint=self.embedding_fingerprint, trace=trace,
        )


def retrieval_metrics(result: V2RetrievalResult, row: Mapping) -> dict:
    gold = tuple(row.get("gold_evidence_spans", ()))
    values = {
        "answerable": float(row.get("gold_answerability") == "ANSWERABLE"),
        "final_k": float(result.final_k), "pre_dedup_k": float(result.pre_dedup_k),
        "post_dedup_k": float(result.post_dedup_k), "context_tokens": float(result.context_tokens),
        "candidates_examined": float(result.candidates_examined),
        "retrieval_rounds": float(result.retrieval_rounds),
    }
    if not gold:
        values.update(evidence_recall=float("nan"), evidence_precision=float("nan"),
                      complete_evidence=float("nan"), mrr=float("nan"), ndcg=float("nan"))
        return values
    covered, relevant = _covered_spans(result.final_evidence, gold)
    values["evidence_recall"] = len(covered) / len(gold)
    values["evidence_precision"] = sum(relevant) / len(relevant) if relevant else 0.0
    values["complete_evidence"] = float(len(covered) == len(gold))
    values["mrr"] = next((1.0 / rank for rank, flag in enumerate(relevant, 1) if flag), 0.0)
    dcg = sum(flag / math.log2(rank + 1) for rank, flag in enumerate(relevant, 1))
    ideal_count = min(len(gold), len(relevant))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    values["ndcg"] = dcg / idcg if idcg else 0.0
    return values


def manifest_payload(config: AdaptiveSingleViewV2Config, input_path: Path,
                     embedding_fingerprint: str, protocol_path: Path) -> dict:
    code_path = Path(__file__)
    return {
        "status": "RETRIEVAL_ONLY", "method": "adaptive_single_view_v2",
        "configuration": asdict(config), "configuration_fingerprint": config.fingerprint,
        "embedding_fingerprint": embedding_fingerprint,
        "input_path": str(input_path.resolve()), "input_sha256": file_sha256(input_path),
        "protocol_path": str(protocol_path.resolve()), "protocol_sha256": file_sha256(protocol_path),
        "code_sha256": file_sha256(code_path), "python": sys.version,
        "platform": platform.platform(), "llm_calls_performed": 0,
    }


def validate_resume(existing: Mapping, expected: Mapping) -> None:
    fields = ("method", "configuration_fingerprint", "embedding_fingerprint", "input_sha256",
              "protocol_sha256", "code_sha256")
    mismatches = [field for field in fields if existing.get(field) != expected.get(field)]
    if mismatches:
        raise ValueError(f"resume fingerprint mismatch: {', '.join(mismatches)}")


def _json_safe(value: object) -> object:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def result_record(example_id: str, result: V2RetrievalResult, metrics: dict,
                  elapsed_ms: float, partition: str | None = None) -> dict:
    return _json_safe({
        "example_id": example_id, "design_partition": partition,
        "representation": result.representation, "metrics": metrics,
        "candidates_examined": result.candidates_examined,
        "retrieval_rounds": result.retrieval_rounds, "pre_dedup_k": result.pre_dedup_k,
        "post_dedup_k": result.post_dedup_k, "final_k": result.final_k,
        "context_tokens": result.context_tokens, "stop_reason": result.stop_reason,
        "thresholds": result.thresholds,
        "configuration_fingerprint": result.configuration_fingerprint,
        "embedding_fingerprint": result.embedding_fingerprint,
        "latency_ms": elapsed_ms,
        "evidence": [{"chunk_id": item.chunk.chunk_id, "rank": item.candidate.rank,
                      "canonical_unit_ids": list(item.chunk.canonical_unit_ids),
                      "text": item.chunk.evidence_text} for item in result.final_evidence],
        "trace": result.trace,
    })


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run retrieval-only Adaptive Single-View V2.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--protocol", type=Path,
                        default=Path("output/adaptive_single_view_v2/PROTOCOL.json"))
    parser.add_argument("--representation", required=True, choices=REPRESENTATIONS)
    parser.add_argument("--max-context-tokens", required=True, type=int,
                        choices=(*TOKEN_OPERATING_POINTS, TOKEN_SAFETY_MAXIMUM))
    parser.add_argument("--relevance-threshold", type=float, default=0.55)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=60)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--max-chunks", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = AdaptiveSingleViewV2Config(
        representation=args.representation, max_context_tokens=args.max_context_tokens,
        relevance_percentile_threshold=args.relevance_threshold,
        empty_batch_patience=args.patience, max_candidates_examined=args.max_candidates,
        max_retrieval_rounds=args.max_rounds, max_final_chunks=args.max_chunks, seed=args.seed,
    )
    rows = _load_jsonl(args.input)
    if any(":train:" not in row["example_id"] for row in rows):
        raise ValueError("v2 development CLI accepts training IDs only")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "RUN_MANIFEST.json"
    results_path = args.output_dir / "PER_EXAMPLE.jsonl"
    completed: set[str] = set()
    systems: dict[str, AdaptiveSingleViewV2System] = {}

    # Construct the first corpus before freezing the run manifest so the exact
    # embedding fingerprint is measured, not inferred from configuration text.
    first_documents = _documents(rows[0]) if rows else []
    first_system = AdaptiveSingleViewV2System.research(first_documents, config) if rows else None
    embedding_fp = first_system.embedding_fingerprint if first_system else canonical_sha256({
        "model": config.embedding_model, "revision": config.embedding_revision,
    })
    expected = manifest_payload(config, args.input, embedding_fp, args.protocol)
    if args.resume:
        if not manifest_path.exists() or not results_path.exists():
            raise ValueError("resume requested but manifest/results are missing")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_resume(existing, expected)
        with results_path.open(encoding="utf-8") as stream:
            completed = {json.loads(line)["example_id"] for line in stream if line.strip()}
    elif manifest_path.exists() or results_path.exists():
        raise FileExistsError("output exists; choose a new versioned directory or use --resume")
    else:
        manifest_path.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")

    for position, row in enumerate(rows):
        if row["example_id"] in completed:
            continue
        documents = _documents(row)
        corpus_key = canonical_sha256([asdict(document) for document in documents])
        if corpus_key not in systems:
            systems[corpus_key] = first_system if position == 0 and first_system is not None else (
                AdaptiveSingleViewV2System.research(documents, config))
        started = time.perf_counter()
        result = systems[corpus_key].retrieve(row["query"])
        elapsed_ms = (time.perf_counter() - started) * 1000
        record = result_record(row["example_id"], result, retrieval_metrics(result, row), elapsed_ms,
                               row.get("design_partition"))
        with results_path.open("a", encoding="utf-8", newline="") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
