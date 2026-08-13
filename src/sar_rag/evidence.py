from __future__ import annotations

from collections import defaultdict

from .models import EvidenceItem, Finding, Requirement


def requirement_statuses(requirements: tuple[Requirement, ...], evidence: list[EvidenceItem]) -> dict[str, str]:
    statuses: dict[str, set[str]] = defaultdict(set)
    for item in evidence:
        for finding in item.judgment.findings:
            if finding.status != "UNRESOLVED":
                statuses[finding.requirement_id].add(finding.status)
    result = {}
    for requirement in requirements:
        values = statuses.get(requirement.requirement_id, set())
        if not values:
            result[requirement.requirement_id] = "UNRESOLVED"
        elif "SUPPORTED" in values and "REFUTED" in values:
            result[requirement.requirement_id] = "CONFLICT"
        elif len(values) == 1:
            result[requirement.requirement_id] = next(iter(values))
        else:
            result[requirement.requirement_id] = "CONFLICT"
    return result


def applicability_coverage(requirements: tuple[Requirement, ...], evidence: list[EvidenceItem]) -> dict[str, set[str]]:
    """Return requested applicability-variable names that are actually evidenced."""
    requested = {
        requirement.requirement_id: {name.strip().casefold() for name in requirement.applicability_variables}
        for requirement in requirements
    }
    covered: dict[str, set[str]] = {requirement.requirement_id: set() for requirement in requirements}
    for item in evidence:
        for finding in item.judgment.findings:
            if finding.status == "UNRESOLVED" or finding.requirement_id not in covered:
                continue
            for name, values in finding.applicability:
                normalized_name = name.strip().casefold()
                if normalized_name in requested[finding.requirement_id] and any(value.strip() for value in values):
                    covered[finding.requirement_id].add(normalized_name)
    return covered


def missing_applicability(requirements: tuple[Requirement, ...], evidence: list[EvidenceItem]) -> dict[str, set[str]]:
    coverage = applicability_coverage(requirements, evidence)
    return {
        requirement.requirement_id: (
            {name.strip().casefold() for name in requirement.applicability_variables}
            - coverage[requirement.requirement_id]
        )
        for requirement in requirements
    }


def exact_merge(items: list[EvidenceItem]) -> list[EvidenceItem]:
    merged: dict[tuple, EvidenceItem] = {}
    for item in items:
        key = (item.candidate.version_id, tuple(sorted(item.candidate.canonical_unit_ids)))
        if key not in merged:
            item.provenance = [(item.candidate.representation_id, item.candidate.rank)]
            merged[key] = item
        else:
            merged[key].provenance.append((item.candidate.representation_id, item.candidate.rank))
            merged[key].obligations.update(item.obligations)
    return list(merged.values())


def _applicability_signature(item: EvidenceItem) -> tuple:
    return tuple(sorted(
        (finding.requirement_id, finding.applicability)
        for finding in item.judgment.findings if finding.status != "UNRESOLVED"
    ))


def _claim_signature(item: EvidenceItem) -> tuple:
    return tuple(sorted(
        (f.requirement_id, f.status, " ".join((f.claim or "").lower().split()), tuple(sorted(q.key for q in f.qualifiers)))
        for f in item.judgment.findings if f.status != "UNRESOLVED"
    ))


def deduplicate_after_retrieval(items: list[EvidenceItem], required_obligations: set[str]) -> list[EvidenceItem]:
    """Exact, containment, then conservative semantic deduplication."""
    retained = exact_merge(items)

    changed = True
    while changed:
        changed = False
        for i in range(len(retained)):
            for j in range(i + 1, len(retained)):
                a, b = retained[i], retained[j]
                if (a.candidate.document_id, a.candidate.version_id) != (b.candidate.document_id, b.candidate.version_id):
                    continue
                ua, ub = set(a.candidate.canonical_unit_ids), set(b.candidate.canonical_unit_ids)
                same_scope = _applicability_signature(a) == _applicability_signature(b)
                if not same_scope:
                    continue
                remove = None
                if ua <= ub and a.obligations <= b.obligations:
                    remove = i
                elif ub <= ua and b.obligations <= a.obligations:
                    remove = j
                elif _claim_signature(a) == _claim_signature(b) and a.obligations == b.obligations:
                    remove = i if a.chunk.word_count >= b.chunk.word_count else j
                if remove is not None:
                    trial = [x for k, x in enumerate(retained) if k != remove]
                    covered = set().union(*(x.obligations for x in trial)) if trial else set()
                    if required_obligations <= covered:
                        retained.pop(remove)
                        changed = True
                        break
            if changed:
                break
    return retained


def compact(items: list[EvidenceItem], obligations: set[str], word_budget: int) -> list[EvidenceItem]:
    uncovered = set(obligations)
    remaining = list(items)
    selected: list[EvidenceItem] = []
    used_words = 0
    while uncovered:
        viable = [x for x in remaining if x.obligations & uncovered]
        if not viable:
            break
        viable.sort(key=lambda x: (
            -(len(x.obligations & uncovered) / max(x.chunk.word_count, 1)),
            x.chunk.word_count,
            x.candidate.candidate_id,
        ))
        best = viable[0]
        if used_words + best.chunk.word_count > word_budget:
            break
        selected.append(best)
        remaining.remove(best)
        used_words += best.chunk.word_count
        uncovered -= best.obligations

    for item in sorted(selected, key=lambda x: -x.chunk.word_count):
        trial = [x for x in selected if x is not item]
        covered = set().union(*(x.obligations for x in trial)) if trial else set()
        if obligations <= covered:
            selected.remove(item)
    return selected
