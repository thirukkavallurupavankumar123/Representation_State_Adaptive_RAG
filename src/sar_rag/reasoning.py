from __future__ import annotations

from hashlib import sha256
import re
from typing import Callable, Protocol

from .models import Candidate, Chunk, Finding, Judgment, QueryAnalysis, Requirement


class RequirementExtractor(Protocol):
    def analyze(self, query: str) -> QueryAnalysis: ...


class EvidenceJudge(Protocol):
    def judge(self, query: QueryAnalysis, candidate: Candidate, chunk: Chunk) -> Judgment: ...


class AnswerGenerator(Protocol):
    def generate(self, query: QueryAnalysis, evidence: list, unresolved: tuple[str, ...], answerability: str) -> str: ...


class RequirementReformulator(Protocol):
    def reformulate(self, query: QueryAnalysis, requirement: Requirement) -> str: ...


class WholeQueryExtractor:
    """Safe local fallback: never invents requirements."""

    def analyze(self, query: str) -> QueryAnalysis:
        query_id = sha256(query.encode()).hexdigest()[:24]
        return QueryAnalysis(query_id, query, (Requirement("R1", query, True),), (), False)


class KeywordJudge:
    """Smoke-test-only lexical judge, not valid for research results."""

    STOP = {"the", "a", "an", "is", "are", "was", "were", "can", "could", "what", "who", "this", "that", "to", "of", "in", "on", "for", "and", "or"}

    def judge(self, query: QueryAnalysis, candidate: Candidate, chunk: Chunk) -> Judgment:
        text_tokens = set(re.findall(r"[a-z0-9]+", chunk.evidence_text.lower())) - self.STOP
        findings = []
        for requirement in query.requirements:
            req_tokens = set(re.findall(r"[a-z0-9]+", requirement.text.lower())) - self.STOP
            overlap = len(text_tokens & req_tokens) / max(len(req_tokens), 1)
            status = "SUPPORTED" if overlap >= 0.25 else "UNRESOLVED"
            findings.append(Finding(
                requirement.requirement_id, status,
                chunk.evidence_text if status == "SUPPORTED" else None,
                candidate.canonical_unit_ids if status == "SUPPORTED" else (),
            ))
        return Judgment(candidate.candidate_id, tuple(findings), valid=True)


class SimpleReformulator:
    def reformulate(self, query: QueryAnalysis, requirement: Requirement) -> str:
        return requirement.text


class EvidenceOnlyAnswerGenerator:
    """Deterministic fallback that exposes evidence instead of fabricating prose."""

    def generate(self, query: QueryAnalysis, evidence: list, unresolved: tuple[str, ...], answerability: str) -> str:
        lines = [f"Answerability: {answerability}."]
        for index, item in enumerate(evidence, 1):
            lines.append(f"[{index}] {item.chunk.evidence_text}")
        if unresolved:
            lines.append("Unresolved requirements: " + "; ".join(unresolved))
        return "\n".join(lines)


class ScriptedRequirementExtractor:
    def __init__(self, analysis: QueryAnalysis):
        self.analysis = analysis

    def analyze(self, query: str) -> QueryAnalysis:
        if query != self.analysis.original_query:
            raise ValueError("scripted analysis query mismatch")
        return self.analysis


class ScriptedJudge:
    """Deterministic oracle used only for control-flow tests."""

    def __init__(self, resolver: Callable[[QueryAnalysis, Candidate, Chunk], Judgment]):
        self.resolver = resolver

    def judge(self, query: QueryAnalysis, candidate: Candidate, chunk: Chunk) -> Judgment:
        return self.resolver(query, candidate, chunk)
