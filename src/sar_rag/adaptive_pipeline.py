from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np

from .config import SarRagConfig
from .corpus import CorpusArtifacts, build_corpus
from .encoding import Encoder, SentenceTransformerEncoder
from .evidence import exact_merge
from .groq import GroqClient, GroqOneCallAnswerGenerator
from .models import (
    Candidate, Document, EvidenceItem, FinalAnswerDecision, Judgment, PipelineResult,
    QueryAnalysis, Requirement,
)
from .selection import align_candidates, select_rank_only_representations


class FinalAnswerer(Protocol):
    def generate(self, query: str, evidence: list[EvidenceItem]) -> FinalAnswerDecision: ...


class SarRagSystem:
    """Rank-only representation selection and one-LLM-call adaptive RAG."""

    def __init__(self, artifacts: CorpusArtifacts, encoder: Encoder,
                 answerer: FinalAnswerer, config: SarRagConfig | None = None):
        self.artifacts = artifacts
        self.encoder = encoder
        self.answerer = answerer
        self.config = config or SarRagConfig()
        self.run_metadata: dict = {}

    @classmethod
    def research(cls, documents: Iterable[Document], config: SarRagConfig | None = None,
                 embedding_revision: str | None = None,
                 groq_api_key: str | None = None) -> "SarRagSystem":
        config = config or SarRagConfig()
        local_path = Path(config.local_embedding_path) if config.local_embedding_path else None
        if local_path is not None and not local_path.is_absolute():
            local_path = Path(__file__).resolve().parents[2] / local_path
        embedding_source = str(local_path.resolve()) if local_path and local_path.is_dir() else config.embedding_model
        resolved_revision = embedding_revision or config.embedding_revision
        encoder = SentenceTransformerEncoder(embedding_source, revision=resolved_revision)
        artifacts = build_corpus(documents, encoder, config.enabled_representations)
        client = GroqClient(config.groq_model, api_key=groq_api_key)
        system = cls(artifacts, encoder, GroqOneCallAnswerGenerator(client), config)
        system.run_metadata = {
            "method": "rank_only_representation_adaptive_rag_one_llm_call",
            "embedding_model_id": config.embedding_model,
            "embedding_source": embedding_source,
            "embedding_revision": resolved_revision,
            "embedding_dimension": encoder.dimension,
            "groq_model": config.groq_model,
            "online_llm_call_policy": "exactly_one_final_answer_call",
            "representations": list(config.enabled_representations),
            "probe_depth": config.probe_depth,
            "rrf_k0": config.rrf_k0,
            "max_selected_representations": config.max_selected_representations,
            "representation_marginal_gain_threshold": config.representation_marginal_gain_threshold,
            "adaptive_batch_size": config.adaptive_batch_size,
            "similarity_threshold": config.similarity_threshold,
            "similarity_gap_threshold": config.similarity_gap_threshold,
            "novelty_threshold": config.novelty_threshold,
            "redundancy_threshold": config.redundancy_threshold,
            "marginal_utility_threshold": config.marginal_utility_threshold,
            "empty_batch_patience": config.empty_batch_patience,
            "candidate_budget": config.max_candidates_examined,
            "retrieval_round_budget": config.max_retrieval_rounds,
            "context_token_budget": config.max_final_context_tokens,
            "final_chunk_safety_cap": config.max_final_chunks,
            "threshold_status": "defaults_must_be_frozen_on_training_data_before_confirmatory_evaluation",
        }
        return system

    def _probe(self, query: str) -> list[Candidate]:
        candidates = []
        for representation in self.config.enabled_representations:
            candidates.extend(self.artifacts.indexes[representation].search(
                query, 1, self.config.probe_depth, "probe"
            ))
        return candidates

    def _vector(self, candidate: Candidate) -> np.ndarray:
        return self.artifacts.indexes[candidate.representation_id].vector_for_chunk(candidate.chunk_id)

    @staticmethod
    def _unit_overlap(candidate: Candidate, accepted: list[EvidenceItem]) -> float:
        units = set(candidate.canonical_unit_ids)
        if not units or not accepted:
            return 0.0
        return max(
            len(units & set(item.candidate.canonical_unit_ids)) /
            len(units | set(item.candidate.canonical_unit_ids))
            for item in accepted
        )

    def _semantic_similarity(self, candidate: Candidate, accepted: list[EvidenceItem]) -> float:
        if not accepted:
            return 0.0
        vector = self._vector(candidate)
        return max(float(vector @ self._vector(item.candidate)) for item in accepted)

    def _post_retrieval_deduplicate(
        self, items: list[EvidenceItem], utility: dict[str, float]
    ) -> list[EvidenceItem]:
        retained = exact_merge(items)
        changed = True
        while changed:
            changed = False
            for left in range(len(retained)):
                for right in range(left + 1, len(retained)):
                    a, b = retained[left], retained[right]
                    if (a.candidate.document_id, a.candidate.version_id) != (
                        b.candidate.document_id, b.candidate.version_id
                    ):
                        continue
                    ua, ub = set(a.candidate.canonical_unit_ids), set(b.candidate.canonical_unit_ids)
                    union = ua | ub
                    overlap = len(ua & ub) / len(union) if union else 1.0
                    semantic = float(self._vector(a.candidate) @ self._vector(b.candidate))
                    remove = None
                    if ua < ub:
                        remove = left
                    elif ub < ua:
                        remove = right
                    elif overlap >= 0.8 and semantic >= self.config.redundancy_threshold:
                        score_a = utility.get(a.candidate.candidate_id, 0.0)
                        score_b = utility.get(b.candidate.candidate_id, 0.0)
                        remove = right if (score_a, -a.chunk.word_count) >= (score_b, -b.chunk.word_count) else left
                    if remove is not None:
                        keep = right if remove == left else left
                        retained[keep].provenance.extend(retained[remove].provenance)
                        retained.pop(remove)
                        changed = True
                        break
                if changed:
                    break
        return retained

    def ask(self, query_text: str) -> PipelineResult:
        query = QueryAnalysis(
            sha256(query_text.encode("utf-8")).hexdigest()[:24], query_text,
            (Requirement("R1", query_text),), (), False,
        )
        trace: list[dict] = [{"event": "query_embedded_without_llm"}]
        probe = self._probe(query_text)
        groups = align_candidates(probe)
        selected, profiles, group_weights = select_rank_only_representations(
            self.config.enabled_representations, groups, self.config.rrf_k0,
            self.config.max_selected_representations,
            self.config.representation_marginal_gain_threshold,
        )
        if self.config.representation_selection_mode == "all":
            selected = set(self.config.enabled_representations)
            for profile in profiles.values():
                profile.selected = True
                profile.selection_reasons.append("ablation_all_representations")
        trace.extend([
            {
                "event": "probe_completed_without_llm",
                "probe_depth": self.config.probe_depth,
                "candidate_count": len(probe),
            },
            {
                "event": "normalized_rrf_completed",
                "rrf_k0": self.config.rrf_k0,
                "group_weights": group_weights,
            },
            {
                "event": "representations_selected_without_llm",
                "selected": sorted(selected),
                "profiles": {
                    name: {
                        "coverage": profile.normalized_coverage_score,
                        "marginal_gain": profile.marginal_gain,
                    } for name, profile in profiles.items()
                },
            },
        ])

        probe_by_rep = {
            rep: sorted((candidate for candidate in probe if candidate.representation_id == rep),
                        key=lambda candidate: candidate.rank)
            for rep in self.config.enabled_representations
        }
        accepted: list[EvidenceItem] = []
        utility: dict[str, float] = {}
        accepted_tokens = 0
        seen_chunks: set[str] = set()
        candidates_examined = len(probe)

        def examine(candidate: Candidate) -> bool:
            nonlocal accepted_tokens
            if candidate.chunk_id in seen_chunks:
                return False
            seen_chunks.add(candidate.chunk_id)
            chunk = self.artifacts.chunks_by_id[candidate.chunk_id]
            index = self.artifacts.indexes[candidate.representation_id]
            relevance = index.relevance_percentile(query_text, candidate.raw_score)
            unit_overlap = self._unit_overlap(candidate, accepted)
            semantic_similarity = self._semantic_similarity(candidate, accepted)
            novelty = (
                float(np.clip(1.0 - unit_overlap * max(semantic_similarity, 0.0), 0.0, 1.0))
                if self.config.novelty_enabled else 1.0
            )
            redundant = unit_overlap >= 0.8 and semantic_similarity >= self.config.redundancy_threshold
            marginal_utility = relevance * novelty
            tokens = self.encoder.token_count(chunk.evidence_text)
            within_budget = (
                accepted_tokens + tokens <= self.config.max_final_context_tokens
                and len(accepted) < self.config.max_final_chunks
            )
            passes = (
                relevance >= self.config.similarity_threshold
                and novelty >= self.config.novelty_threshold
                and marginal_utility >= self.config.marginal_utility_threshold
                and not redundant
                and within_budget
            )
            trace.append({
                "event": "candidate_scored_without_llm",
                "candidate_id": candidate.candidate_id,
                "representation": candidate.representation_id,
                "rank": candidate.rank,
                "raw_cosine": candidate.raw_score,
                "normalized_relevance": relevance,
                "canonical_overlap": unit_overlap,
                "semantic_similarity_to_selected": semantic_similarity,
                "novelty": novelty,
                "marginal_utility": marginal_utility,
                "tokens": tokens,
                "accepted": passes,
                "rejection_reason": None if passes else (
                    "similarity_percentile" if relevance < self.config.similarity_threshold else
                    "novelty" if novelty < self.config.novelty_threshold else
                    "marginal_utility" if marginal_utility < self.config.marginal_utility_threshold else
                    "redundancy" if redundant else "context_or_chunk_budget"
                ),
            })
            if not passes:
                return False
            item = EvidenceItem(
                candidate, chunk, Judgment(candidate.candidate_id, ()),
                {f"UNIT:{unit_id}" for unit_id in candidate.canonical_unit_ids},
                [(candidate.representation_id, candidate.rank)],
            )
            accepted.append(item)
            utility[candidate.candidate_id] = marginal_utility
            accepted_tokens += tokens
            return True

        for representation in sorted(selected):
            for candidate in probe_by_rep[representation]:
                examine(candidate)

        next_rank = {rep: self.config.probe_depth + 1 for rep in selected}
        empty_batches = {rep: 0 for rep in selected}
        active = set(selected)
        previous_score = {
            rep: probe_by_rep[rep][-1].raw_score if probe_by_rep[rep] else 1.0
            for rep in selected
        }
        retrieval_round = 0
        stop_reason = (
            "probe_only_ablation" if not self.config.adaptive_retrieval_enabled
            else "no_selected_representation"
        )
        while active and self.config.adaptive_retrieval_enabled:
            if candidates_examined >= self.config.max_candidates_examined:
                stop_reason = "candidate_safety_cap"
                break
            if retrieval_round >= self.config.max_retrieval_rounds:
                stop_reason = "retrieval_round_safety_cap"
                break
            if accepted_tokens >= self.config.max_final_context_tokens:
                stop_reason = "context_token_budget"
                break
            if len(accepted) >= self.config.max_final_chunks:
                stop_reason = "final_chunk_safety_cap"
                break
            retrieval_round += 1
            for representation in sorted(tuple(active)):
                remaining = self.config.max_candidates_examined - candidates_examined
                if remaining <= 0:
                    break
                batch = self.artifacts.indexes[representation].search(
                    query_text, next_rank[representation],
                    min(self.config.adaptive_batch_size, remaining), "adaptive",
                )
                next_rank[representation] += len(batch)
                candidates_examined += len(batch)
                if not batch:
                    active.remove(representation)
                    trace.append({"event": "representation_paused", "representation": representation,
                                  "reason": "index_exhausted"})
                    continue
                added = sum(examine(candidate) for candidate in batch)
                raw_largest_gap = max(
                    [previous_score[representation] - batch[0].raw_score] +
                    [batch[index].raw_score - batch[index + 1].raw_score
                     for index in range(len(batch) - 1)]
                )
                largest_gap = raw_largest_gap / self.artifacts.indexes[representation].score_scale(query_text)
                previous_score[representation] = batch[-1].raw_score
                empty_batches[representation] = 0 if added else empty_batches[representation] + 1
                reason = None
                if empty_batches[representation] >= self.config.empty_batch_patience:
                    reason = "no_marginal_utility_for_patience_batches"
                elif (self.config.similarity_gap_enabled and added == 0
                      and largest_gap >= self.config.similarity_gap_threshold):
                    reason = "similarity_gap_with_no_marginal_utility"
                if reason:
                    active.remove(representation)
                    trace.append({
                        "event": "representation_paused", "representation": representation,
                        "reason": reason, "largest_similarity_gap": largest_gap,
                    })
            if not active:
                stop_reason = "all_selected_representations_paused"

        deduplicated = (
            self._post_retrieval_deduplicate(accepted, utility)
            if self.config.post_retrieval_deduplication_enabled else list(accepted)
        )
        deduplicated.sort(key=lambda item: (
            -utility.get(item.candidate.candidate_id, 0.0),
            item.candidate.rank,
            item.candidate.candidate_id,
        ))
        final_evidence: list[EvidenceItem] = []
        final_tokens = 0
        for item in deduplicated:
            tokens = self.encoder.token_count(item.chunk.evidence_text)
            if final_tokens + tokens > self.config.max_final_context_tokens:
                continue
            if len(final_evidence) >= self.config.max_final_chunks:
                break
            final_evidence.append(item)
            final_tokens += tokens

        # This is intentionally the first and only online LLM operation.
        decision = self.answerer.generate(query_text, final_evidence)
        unresolved = () if decision.answerability == "ANSWERABLE" else ("R1",)
        trace.append({
            "event": "single_final_llm_call_completed",
            "answerability": decision.answerability,
            "cited_evidence_ids": list(decision.cited_evidence_ids),
        })
        trace.append({
            "event": "run_completed",
            "stop_reason": stop_reason,
            "candidates_examined": candidates_examined,
            "retrieved_before_deduplication": len(accepted),
            "final_k": len(final_evidence),
            "final_context_tokens": final_tokens,
            "retrieval_rounds": retrieval_round,
        })
        return PipelineResult(
            query=query,
            selected_representations=tuple(sorted(selected)),
            representation_profiles=profiles,
            evidence_groups=groups,
            provisional_evidence=accepted,
            final_evidence=final_evidence,
            final_k=len(final_evidence),
            answerability=decision.answerability,
            answer=decision.answer,
            unresolved_requirements=unresolved,
            stop_reason=stop_reason,
            trace=trace,
            run_metadata=dict(self.run_metadata),
        )
