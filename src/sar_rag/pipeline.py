from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Iterable

from .config import SarRagConfig
from .corpus import CorpusArtifacts, build_corpus
from .encoding import Encoder, HashingEncoder, SentenceTransformerEncoder
from .evidence import compact, deduplicate_after_retrieval, missing_applicability, requirement_statuses
from .groq import GroqAnswerGenerator, GroqClient, GroqEvidenceJudge, GroqReformulator, GroqRequirementExtractor
from .models import (
    Candidate, Chunk, Document, EvidenceItem, Judgment, PipelineResult,
    QueryAnalysis, RepresentationProfile, RuntimeRepresentationState,
)
from .reasoning import (
    AnswerGenerator, EvidenceJudge, RequirementExtractor, RequirementReformulator,
)
from .selection import align_candidates, build_profiles, score_rrf, select_representations


class LlmJudgedSarRagSystem:
    def __init__(self, artifacts: CorpusArtifacts, extractor: RequirementExtractor,
                 judge: EvidenceJudge, reformulator: RequirementReformulator,
                 answer_generator: AnswerGenerator, config: SarRagConfig | None = None):
        self.artifacts = artifacts
        self.extractor = extractor
        self.judge = judge
        self.reformulator = reformulator
        self.answer_generator = answer_generator
        self.config = config or SarRagConfig()
        self.run_metadata: dict = {}

    @classmethod
    def research(cls, documents: Iterable[Document], config: SarRagConfig | None = None,
                 embedding_revision: str | None = None, groq_api_key: str | None = None) -> "LlmJudgedSarRagSystem":
        """Production research assembly requested by the study protocol."""
        config = config or SarRagConfig()
        local_path = Path(config.local_embedding_path) if config.local_embedding_path else None
        if local_path is not None and not local_path.is_absolute():
            local_path = Path(__file__).resolve().parents[2] / local_path
        embedding_source = str(local_path.resolve()) if local_path and local_path.is_dir() else config.embedding_model
        resolved_revision = embedding_revision or config.embedding_revision
        client = GroqClient(config.groq_model, api_key=groq_api_key)
        encoder = SentenceTransformerEncoder(embedding_source, revision=resolved_revision)
        artifacts = build_corpus(documents, encoder, config.enabled_representations)
        system = cls(
            artifacts,
            GroqRequirementExtractor(client),
            GroqEvidenceJudge(client),
            GroqReformulator(client),
            GroqAnswerGenerator(client),
            config,
        )
        system.run_metadata = {
            "embedding_model_id": config.embedding_model,
            "embedding_source": embedding_source,
            "embedding_revision": resolved_revision,
            "embedding_dimension": encoder.dimension,
            "embedding_pooling": "attention_mask_mean_pooling_then_l2_normalization",
            "groq_model": config.groq_model,
            "probe_depth": config.probe_depth,
            "adaptive_batch_size": config.adaptive_batch_size,
            "candidate_budget": config.max_candidates_examined,
            "retrieval_round_budget": config.max_retrieval_rounds,
            "context_token_budget_legacy_approximation": config.max_final_context_tokens,
        }
        return system

    @classmethod
    def with_components(cls, documents: Iterable[Document], encoder: Encoder,
                        extractor: RequirementExtractor, judge: EvidenceJudge,
                        reformulator: RequirementReformulator,
                        answer_generator: AnswerGenerator,
                        config: SarRagConfig | None = None) -> "LlmJudgedSarRagSystem":
        config = config or SarRagConfig()
        return cls(build_corpus(documents, encoder, config.enabled_representations),
                   extractor, judge, reformulator, answer_generator, config)

    def _retrieve_probe(self, query: str) -> list[Candidate]:
        results = []
        for rep in self.config.enabled_representations:
            results.extend(self.artifacts.indexes[rep].search(
                query, 1, self.config.probe_depth, "probe"
            ))
        return results

    def _expand_context(self, candidate: Candidate) -> Candidate | None:
        """Create the smallest deterministic heading/adjacent expansion."""
        source_units = [self.artifacts.units_by_id[x] for x in candidate.canonical_unit_ids]
        expanded_ids = set(candidate.canonical_unit_ids)
        for unit in source_units:
            for related in (unit.parent_id, unit.previous_unit_id, unit.next_unit_id):
                if related in self.artifacts.units_by_id:
                    related_unit = self.artifacts.units_by_id[related]
                    if (related_unit.document_id, related_unit.version_id) == (candidate.document_id, candidate.version_id):
                        expanded_ids.add(related)
        if expanded_ids == set(candidate.canonical_unit_ids):
            return None
        units = sorted(
            (self.artifacts.units_by_id[x] for x in expanded_ids),
            key=lambda u: (u.character_start, u.character_end),
        )
        chunk_id = sha256(("context_expansion|" + "|".join(u.unit_id for u in units)).encode()).hexdigest()[:24]
        evidence_text = "\n".join(u.raw_text for u in units)
        chunk = Chunk(
            chunk_id=chunk_id,
            representation_id="context_expansion",
            document_id=candidate.document_id,
            version_id=candidate.version_id,
            canonical_unit_ids=tuple(u.unit_id for u in units),
            embedding_text=evidence_text,
            evidence_text=evidence_text,
            word_count=len(evidence_text.split()),
            metadata={"expanded_from": candidate.candidate_id},
        )
        self.artifacts.chunks_by_id[chunk_id] = chunk
        candidate_id = sha256((candidate.candidate_id + "|context_expansion").encode()).hexdigest()[:24]
        return Candidate(
            candidate_id, "context_expansion", chunk_id, candidate.rank, candidate.raw_score,
            candidate.retrieval_query, "adaptive", chunk.canonical_unit_ids,
            candidate.document_id, candidate.version_id, chunk.word_count,
        )

    def ask(self, query_text: str) -> PipelineResult:
        trace: list[dict] = []
        query = self.extractor.analyze(query_text)
        trace.append({"event": "query_analyzed", "requirements": [r.requirement_id for r in query.requirements]})

        # Phase A: Top-p is diagnostic representation selection, not final retrieval k.
        probe = self._retrieve_probe(query.original_query)
        judgments: dict[str, Judgment] = {}
        for candidate in probe:
            chunk = self.artifacts.chunks_by_id[candidate.chunk_id]
            judgments[candidate.candidate_id] = self.judge.judge(query, candidate, chunk)
        groups = align_candidates(probe)
        score_rrf(groups, self.config.rrf_k0)
        profiles = build_profiles(self.config.enabled_representations, groups, judgments, self.config.rrf_k0)
        selected, uncovered_probe = select_representations(profiles)
        trace.extend([
            {"event": "probe_completed", "candidate_count": len(probe), "probe_depth": self.config.probe_depth},
            {"event": "rrf_completed", "groups": {g.group_id: g.rrf_score for g in groups}},
            {"event": "representations_selected", "selected": sorted(selected), "uncovered_probe": sorted(uncovered_probe)},
        ])

        runtime = {
            rep: RuntimeRepresentationState(
                representation_id=rep,
                selected=rep in selected,
                active=rep in selected,
                next_rank=self.config.probe_depth + 1,
            ) for rep in self.config.enabled_representations
        }
        probe_by_rep: dict[str, list[Candidate]] = defaultdict(list)
        for candidate in probe:
            probe_by_rep[candidate.representation_id].append(candidate)

        provisional: list[EvidenceItem] = []
        secured_obligations: set[str] = set()
        # Every probe candidate has already incurred a judge call. Cached judgments are
        # reused below and must not be charged a second time.
        candidates_examined = len(probe)
        processed_candidate_ids: set[str] = set()

        def process(candidate: Candidate, known_judgment: Judgment | None = None,
                    allow_expansion: bool = True) -> bool:
            nonlocal candidates_examined
            already_examined = known_judgment is not None
            if not already_examined and candidates_examined >= self.config.max_candidates_examined:
                return False
            if candidate.candidate_id in processed_candidate_ids:
                return False
            processed_candidate_ids.add(candidate.candidate_id)
            if not already_examined:
                candidates_examined += 1
            chunk = self.artifacts.chunks_by_id[candidate.chunk_id]
            judgment = known_judgment or self.judge.judge(query, candidate, chunk)
            # A request for more context guides routing but is not itself final evidence.
            obligations = judgment.obligation_keys() - set(judgment.context_keys)
            marginal = obligations - secured_obligations
            trace.append({
                "event": "candidate_examined", "candidate_id": candidate.candidate_id,
                "representation": candidate.representation_id, "rank": candidate.rank,
                "stage": candidate.retrieval_stage, "marginal_obligations": sorted(marginal),
            })
            if judgment.valid and marginal:
                item = EvidenceItem(candidate, chunk, judgment, set(obligations),
                                    [(candidate.representation_id, candidate.rank)])
                provisional.append(item)
                secured_obligations.update(obligations)
                trace.append({"event": "candidate_provisionally_accepted", "candidate_id": candidate.candidate_id})
                accepted = True
            else:
                accepted = False
            if allow_expansion and judgment.valid and judgment.context_keys:
                expanded = self._expand_context(candidate)
                if expanded is not None:
                    trace.append({"event": "context_expanded", "from": candidate.candidate_id, "to": expanded.candidate_id})
                    accepted = process(expanded, allow_expansion=False) or accepted
            return accepted

        # Reuse selected probe results as the first candidate stream; do not re-query.
        for rep in sorted(selected):
            for candidate in sorted(probe_by_rep[rep], key=lambda c: c.rank):
                process(candidate, judgments[candidate.candidate_id])

        def statuses() -> dict[str, str]:
            return requirement_statuses(query.requirements, provisional)

        def unresolved_ids() -> tuple[str, ...]:
            current = statuses()
            missing_scope = missing_applicability(query.requirements, provisional)
            return tuple(
                r.requirement_id for r in query.requirements
                if current[r.requirement_id] == "UNRESOLVED" or missing_scope[r.requirement_id]
            )

        def all_critical_resolved() -> bool:
            current = statuses()
            missing_scope = missing_applicability(query.requirements, provisional)
            return all(
                not r.critical or (
                    current[r.requirement_id] in {"SUPPORTED", "REFUTED", "NOT_APPLICABLE", "CONFLICT"}
                    and not missing_scope[r.requirement_id]
                )
                for r in query.requirements
            )

        # Phase B: consume selected indexes incrementally; never compare count to final k.
        retrieval_round = 0
        confirmation_empty = False
        reactivation_done = False
        reformulated: set[str] = set()
        stop_reason = "unknown"

        while True:
            if candidates_examined >= self.config.max_candidates_examined:
                stop_reason = "candidate_budget_exhausted"
                break
            if retrieval_round >= self.config.max_retrieval_rounds:
                stop_reason = "retrieval_round_budget_exhausted"
                break
            if all_critical_resolved() and confirmation_empty:
                stop_reason = "evidence_sufficient_after_empty_confirmation"
                break

            active = [rep for rep, value in runtime.items() if value.active]
            if not active:
                unselected = [rep for rep, value in runtime.items() if not value.selected]
                if unselected and not reactivation_done:
                    reactivation_done = True
                    for rep in unselected:
                        runtime[rep].selected = runtime[rep].active = runtime[rep].reactivated = True
                        for candidate in sorted(probe_by_rep[rep], key=lambda c: c.rank):
                            process(candidate, judgments[candidate.candidate_id])
                    trace.append({"event": "representations_reactivated", "representations": unselected})
                    continue

                unresolved = unresolved_ids()
                pending = [r for r in query.requirements if r.requirement_id in unresolved and r.requirement_id not in reformulated]
                if pending:
                    requirement = pending[0]
                    reformulated.add(requirement.requirement_id)
                    rewritten = self.reformulator.reformulate(query, requirement)
                    gained = False
                    for rep in self.config.enabled_representations:
                        for candidate in self.artifacts.indexes[rep].search(
                            rewritten, 1, self.config.adaptive_batch_size, "reformulation"
                        ):
                            gained = process(candidate) or gained
                    trace.append({"event": "requirement_reformulated", "requirement": requirement.requirement_id, "gained": gained})
                    continue
                stop_reason = "unresolved_no_active_representation"
                break

            retrieval_round += 1
            before_round = len(secured_obligations)
            was_confirmation = all_critical_resolved()
            for rep in list(active):
                rep_state = runtime[rep]
                batch = self.artifacts.indexes[rep].search(
                    query.original_query,
                    rep_state.next_rank,
                    self.config.adaptive_batch_size,
                    "adaptive",
                )
                rep_state.next_rank += len(batch)
                gained = False
                for candidate in batch:
                    gained = process(candidate) or gained
                if not gained:
                    rep_state.active = False
                    rep_state.consecutive_empty_batches += 1
                    rep_state.pause_reason = "latest_batch_added_no_value"
                    trace.append({"event": "representation_paused", "representation": rep})

            added = len(secured_obligations) > before_round
            if was_confirmation:
                confirmation_empty = not added
                if added:
                    # A new qualifier/context obligation requires another empty confirmation.
                    for rep in selected:
                        if runtime[rep].next_rank <= len(self.artifacts.indexes[rep].chunks):
                            runtime[rep].active = True

        # Phase C: final deduplication happens only after adaptive retrieval stops.
        required_obligations = set().union(*(item.obligations for item in provisional)) if provisional else set()
        deduplicated = deduplicate_after_retrieval(provisional, required_obligations)
        final_evidence = compact(deduplicated, required_obligations, self.config.max_final_context_tokens)
        final_covered = set().union(*(x.obligations for x in final_evidence)) if final_evidence else set()
        compaction_incomplete = not required_obligations <= final_covered
        if compaction_incomplete:
            trace.append({
                "event": "context_budget_insufficient",
                "missing_obligations": sorted(required_obligations - final_covered),
            })

        current_status = requirement_statuses(query.requirements, final_evidence)
        missing_scope = missing_applicability(query.requirements, final_evidence)
        unresolved = tuple(
            r.requirement_id for r in query.requirements
            if current_status[r.requirement_id] == "UNRESOLVED" or missing_scope[r.requirement_id]
        )
        conflicts = tuple(r.requirement_id for r in query.requirements if current_status[r.requirement_id] == "CONFLICT")
        resolved_count = sum(v in {"SUPPORTED", "REFUTED", "NOT_APPLICABLE"} for v in current_status.values())
        if compaction_incomplete:
            answerability = "INSUFFICIENT_UNDER_BUDGET"
        elif conflicts and not unresolved:
            answerability = "ANSWERABLE_WITH_CONFLICT"
        elif not unresolved:
            answerability = "ANSWERABLE"
        elif resolved_count:
            answerability = "PARTIALLY_ANSWERABLE"
        elif "budget_exhausted" in stop_reason:
            answerability = "INSUFFICIENT_UNDER_BUDGET"
        else:
            answerability = "INSUFFICIENT"

        answer = self.answer_generator.generate(query, final_evidence, unresolved, answerability)
        trace.append({
            "event": "run_completed", "stop_reason": stop_reason,
            "provisional_count": len(provisional), "final_k": len(final_evidence),
            "answerability": answerability, "candidates_examined": candidates_examined,
            "missing_applicability": {key: sorted(value) for key, value in missing_scope.items() if value},
        })
        return PipelineResult(
            query=query,
            selected_representations=tuple(sorted(selected)),
            representation_profiles=profiles,
            evidence_groups=groups,
            provisional_evidence=provisional,
            final_evidence=final_evidence,
            final_k=len(final_evidence),
            answerability=answerability,
            answer=answer,
            unresolved_requirements=unresolved,
            stop_reason=stop_reason,
            trace=trace,
            run_metadata=dict(self.run_metadata),
        )


# Public primary system: deterministic retrieval controls and one final LLM call.
# The class above is retained only for the explicitly named legacy ablation.
from .adaptive_pipeline import SarRagSystem  # noqa: E402
