from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


FindingStatus = Literal["SUPPORTED", "REFUTED", "NOT_APPLICABLE", "UNRESOLVED"]


@dataclass(frozen=True)
class Document:
    document_id: str
    version_id: str
    title: str
    text: str
    effective_date: str | None = None
    authority_level: str | None = None
    supersedes_version_id: str | None = None


@dataclass(frozen=True)
class CanonicalUnit:
    unit_id: str
    document_id: str
    version_id: str
    unit_type: str
    parent_id: str | None
    previous_unit_id: str | None
    next_unit_id: str | None
    section_path: tuple[str, ...]
    character_start: int
    character_end: int
    raw_text: str
    normalized_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    representation_id: str
    document_id: str
    version_id: str
    canonical_unit_ids: tuple[str, ...]
    embedding_text: str
    evidence_text: str
    word_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    representation_id: str
    chunk_id: str
    rank: int
    raw_score: float
    retrieval_query: str
    retrieval_stage: Literal["probe", "adaptive", "reformulation"]
    canonical_unit_ids: tuple[str, ...]
    document_id: str
    version_id: str
    word_count: int


@dataclass(frozen=True)
class Requirement:
    requirement_id: str
    text: str
    critical: bool = True
    entities: tuple[str, ...] = ()
    applicability_variables: tuple[str, ...] = ()


@dataclass(frozen=True)
class QueryFact:
    fact_id: str
    text: str


@dataclass(frozen=True)
class QueryAnalysis:
    query_id: str
    original_query: str
    requirements: tuple[Requirement, ...]
    query_facts: tuple[QueryFact, ...] = ()
    decomposition_confident: bool = True


@dataclass(frozen=True)
class Qualifier:
    qualifier_type: str
    text: str
    cited_unit_ids: tuple[str, ...]

    @property
    def key(self) -> str:
        return f"QUALIFIER:{self.qualifier_type}:{' '.join(self.text.lower().split())}"


@dataclass(frozen=True)
class Finding:
    requirement_id: str
    status: FindingStatus
    claim: str | None
    cited_unit_ids: tuple[str, ...]
    applicability: tuple[tuple[str, tuple[str, ...]], ...] = ()
    qualifiers: tuple[Qualifier, ...] = ()

    def obligation_keys(self) -> set[str]:
        if self.status == "UNRESOLVED":
            return set()
        keys = {f"REQ:{self.requirement_id}:{self.status}"}
        for name, values in self.applicability:
            for value in values:
                keys.add(f"APPLICABILITY:{name}:{value.lower()}")
        keys.update(f"{q.key}:REQ:{self.requirement_id}" for q in self.qualifiers)
        return keys


@dataclass(frozen=True)
class Judgment:
    candidate_id: str
    findings: tuple[Finding, ...]
    context_keys: tuple[str, ...] = ()
    contradiction_keys: tuple[str, ...] = ()
    independent_source_keys: tuple[str, ...] = ()
    valid: bool = True

    def obligation_keys(self) -> set[str]:
        if not self.valid:
            return set()
        keys: set[str] = set(self.context_keys)
        keys.update(self.contradiction_keys)
        keys.update(self.independent_source_keys)
        for finding in self.findings:
            keys.update(finding.obligation_keys())
        return keys


@dataclass
class EvidenceGroup:
    group_id: str
    anchor_candidate_id: str
    document_id: str
    version_id: str
    members: list[Candidate]
    rrf_score: float = 0.0


@dataclass
class RepresentationProfile:
    representation_id: str
    obligations: set[str] = field(default_factory=set)
    exclusive_obligations: set[str] = field(default_factory=set)
    useful_group_ids: set[str] = field(default_factory=set)
    redundant_group_ids: set[str] = field(default_factory=set)
    rrf_contribution: float = 0.0
    normalized_coverage_score: float = 0.0
    marginal_gain: float = 0.0
    mean_probe_words: float = 0.0
    selected: bool = False
    selection_reasons: list[str] = field(default_factory=list)


@dataclass
class RuntimeRepresentationState:
    representation_id: str
    selected: bool
    active: bool
    next_rank: int
    consecutive_empty_batches: int = 0
    pause_reason: str | None = None
    reactivated: bool = False


@dataclass
class EvidenceItem:
    candidate: Candidate
    chunk: Chunk
    judgment: Judgment
    obligations: set[str]
    provenance: list[tuple[str, int]] = field(default_factory=list)


@dataclass
class PipelineResult:
    query: QueryAnalysis
    selected_representations: tuple[str, ...]
    representation_profiles: dict[str, RepresentationProfile]
    evidence_groups: list[EvidenceGroup]
    provisional_evidence: list[EvidenceItem]
    final_evidence: list[EvidenceItem]
    final_k: int
    answerability: str
    answer: str
    unresolved_requirements: tuple[str, ...]
    stop_reason: str
    trace: list[dict[str, Any]]
    run_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FinalAnswerDecision:
    answer: str
    answerability: Literal["ANSWERABLE", "PARTIALLY_ANSWERABLE", "INSUFFICIENT"]
    cited_evidence_ids: tuple[str, ...] = ()
