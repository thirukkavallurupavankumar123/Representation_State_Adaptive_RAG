from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from .encoding import Encoder
from .groq import GroqClient, GroqOneCallAnswerGenerator
from .models import (
    EvidenceItem, Finding, Judgment, PipelineResult, QueryAnalysis, Requirement,
)
from .pipeline import CorpusArtifacts, build_corpus


@dataclass(frozen=True)
class TraditionalRagConfig:
    static_k: int = 5
    max_context_tokens: int = 2_000
    representation: str = "sliding_window"

    def __post_init__(self) -> None:
        if self.static_k < 1 or self.max_context_tokens < 1:
            raise ValueError("Traditional RAG budgets must be positive")


class TraditionalRagSystem:
    """Single dense index, static top-k, no probe, routing, judge, or adaptive stopping."""

    def __init__(self, artifacts: CorpusArtifacts, answerer: GroqOneCallAnswerGenerator,
                 config: TraditionalRagConfig | None = None):
        self.artifacts = artifacts
        self.answerer = answerer
        self.config = config or TraditionalRagConfig()

    @classmethod
    def build(cls, documents, encoder: Encoder, client: GroqClient,
              config: TraditionalRagConfig | None = None) -> "TraditionalRagSystem":
        config = config or TraditionalRagConfig()
        artifacts = build_corpus(documents, encoder, (config.representation,))
        return cls(artifacts, GroqOneCallAnswerGenerator(client), config)

    def ask(self, query_text: str) -> PipelineResult:
        candidates = self.artifacts.indexes[self.config.representation].search(
            query_text, 1, self.config.static_k, "adaptive"
        )
        evidence: list[EvidenceItem] = []
        used_tokens = 0
        for candidate in candidates:
            chunk = self.artifacts.chunks_by_id[candidate.chunk_id]
            tokens = self.artifacts.indexes[self.config.representation].encoder.token_count(
                chunk.evidence_text
            )
            if used_tokens + tokens > self.config.max_context_tokens:
                continue
            judgment = Judgment(candidate.candidate_id, (), valid=True)
            evidence.append(EvidenceItem(
                candidate, chunk, judgment, set(),
                [(candidate.representation_id, candidate.rank)],
            ))
            used_tokens += tokens

        decision = self.answerer.generate(query_text, evidence)
        answerability = decision.answerability
        analysis = QueryAnalysis(
            query_id="traditional:" + sha256(query_text.encode("utf-8")).hexdigest()[:20],
            original_query=query_text,
            requirements=(Requirement("R1", query_text),),
            decomposition_confident=False,
        )
        unresolved = ("R1",) if answerability == "INSUFFICIENT" else ()
        return PipelineResult(
            query=analysis,
            selected_representations=(self.config.representation,),
            representation_profiles={},
            evidence_groups=[],
            provisional_evidence=list(evidence),
            final_evidence=evidence,
            final_k=len(evidence),
            answerability=answerability,
            answer=decision.answer,
            unresolved_requirements=unresolved,
            stop_reason="static_top_k",
            trace=[{
                # `candidates_examined` must report the candidates this method actually
                # scored (the static top-k it requested), not the smaller number that
                # survived the shared context-token budget.  Reporting final_k here
                # understated the baseline's retrieval work in the efficiency comparison.
                "event": "run_completed", "retrieved_k": len(candidates),
                "candidates_examined": len(candidates),
                "retrieval_rounds": 1,
                "retrieved_before_deduplication": len(evidence),
                "final_k": len(evidence), "context_tokens": used_tokens,
                "final_context_tokens": used_tokens,
            }],
            run_metadata={
                "method": "traditional_dense_static_rag",
                "representation": self.config.representation,
                "static_k": self.config.static_k,
                "context_token_budget": self.config.max_context_tokens,
            },
        )
