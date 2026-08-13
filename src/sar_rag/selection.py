from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import math

from .models import Candidate, EvidenceGroup, Judgment, RepresentationProfile


def compatible(a: Candidate, b: Candidate) -> bool:
    if (a.document_id, a.version_id) != (b.document_id, b.version_id):
        return False
    ua, ub = set(a.canonical_unit_ids), set(b.canonical_unit_ids)
    if ua == ub or ua <= ub or ub <= ua:
        return True
    return bool(ua or ub) and len(ua & ub) / len(ua | ub) >= 0.5


def align_candidates(candidates: list[Candidate]) -> list[EvidenceGroup]:
    """Anchor-based alignment; retains every candidate and provenance."""
    ordered = sorted(candidates, key=lambda c: (c.rank, c.representation_id, c.candidate_id))
    groups: list[EvidenceGroup] = []
    for candidate in ordered:
        target = next((g for g in groups if compatible(candidate, g.members[0])), None)
        if target is None:
            group_id = sha256(f"group|{candidate.candidate_id}".encode()).hexdigest()[:20]
            groups.append(EvidenceGroup(group_id, candidate.candidate_id, candidate.document_id, candidate.version_id, [candidate]))
        else:
            target.members.append(candidate)
    return groups


def score_rrf(groups: list[EvidenceGroup], k0: int = 60) -> None:
    for group in groups:
        best: dict[str, int] = {}
        for member in group.members:
            best[member.representation_id] = min(member.rank, best.get(member.representation_id, math.inf))
        group.rrf_score = sum(1.0 / (k0 + rank) for rank in best.values())


def select_rank_only_representations(
    representation_ids: tuple[str, ...],
    groups: list[EvidenceGroup],
    k0: int,
    max_selected: int,
    marginal_gain_threshold: float,
) -> tuple[set[str], dict[str, RepresentationProfile], dict[str, float]]:
    """Greedy weighted set cover over normalized RRF evidence groups.

    This selector is deliberately model-free. The first representation maximizes
    total probe-group coverage. A second is selected only when it adds enough RRF
    mass from groups the first representation did not retrieve. Mean probe words
    and representation ID are deterministic tie breakers.
    """
    if not groups:
        return set(), {name: RepresentationProfile(name) for name in representation_ids}, {}
    score_rrf(groups, k0)
    maximum_group_score = len(representation_ids) / (k0 + 1)
    group_weights = {
        group.group_id: group.rrf_score / maximum_group_score
        for group in groups
    }
    total_weight = sum(group_weights.values()) or 1.0
    memberships: dict[str, set[str]] = {name: set() for name in representation_ids}
    words: dict[str, list[int]] = defaultdict(list)
    group_representations: dict[str, set[str]] = {}
    for group in groups:
        reps = {member.representation_id for member in group.members}
        group_representations[group.group_id] = reps
        for member in group.members:
            memberships[member.representation_id].add(group.group_id)
            words[member.representation_id].append(member.word_count)

    profiles = {name: RepresentationProfile(name) for name in representation_ids}
    for name, profile in profiles.items():
        profile.useful_group_ids = set(memberships[name])
        profile.redundant_group_ids = {
            group_id for group_id in memberships[name]
            if len(group_representations[group_id]) > 1
        }
        profile.rrf_contribution = sum(group_weights[group_id] for group_id in memberships[name])
        profile.normalized_coverage_score = profile.rrf_contribution / total_weight
        profile.mean_probe_words = (
            sum(words[name]) / len(words[name]) if words[name] else math.inf
        )

    ordered = sorted(profiles.values(), key=lambda profile: (
        -profile.normalized_coverage_score,
        profile.mean_probe_words,
        profile.representation_id,
    ))
    first = ordered[0]
    first.selected = True
    first.marginal_gain = first.normalized_coverage_score
    first.selection_reasons.append("highest_normalized_rrf_coverage")
    selected = {first.representation_id}
    covered = set(memberships[first.representation_id])

    if max_selected > 1:
        candidates = []
        for profile in ordered[1:]:
            new_groups = memberships[profile.representation_id] - covered
            gain = sum(group_weights[group_id] for group_id in new_groups) / total_weight
            profile.marginal_gain = gain
            candidates.append((profile, new_groups))
        if candidates:
            candidates.sort(key=lambda value: (
                -value[0].marginal_gain,
                value[0].mean_probe_words,
                value[0].representation_id,
            ))
            second, new_groups = candidates[0]
            if second.marginal_gain >= marginal_gain_threshold:
                second.selected = True
                second.selection_reasons.append(
                    "marginal_rrf_gain:" + ",".join(sorted(new_groups))
                )
                selected.add(second.representation_id)
    return selected, profiles, group_weights


def build_profiles(representation_ids: tuple[str, ...], groups: list[EvidenceGroup],
                   judgments: dict[str, Judgment], k0: int) -> dict[str, RepresentationProfile]:
    profiles = {rep: RepresentationProfile(rep) for rep in representation_ids}
    token_lists: dict[str, list[int]] = defaultdict(list)
    obligation_reps: dict[str, set[str]] = defaultdict(set)

    for group in groups:
        useful_group = False
        group_obligations: dict[str, set[str]] = defaultdict(set)
        for member in group.members:
            token_lists[member.representation_id].append(member.word_count)
            judgment = judgments.get(member.candidate_id)
            obligations = judgment.obligation_keys() if judgment else set()
            group_obligations[member.representation_id].update(obligations)
            useful_group = useful_group or bool(obligations)
        for member in group.members:
            profile = profiles[member.representation_id]
            if useful_group and group_obligations[member.representation_id]:
                profile.useful_group_ids.add(group.group_id)
                profile.obligations.update(group_obligations[member.representation_id])
                for key in group_obligations[member.representation_id]:
                    obligation_reps[key].add(member.representation_id)
            else:
                profile.redundant_group_ids.add(group.group_id)

    for key, reps in obligation_reps.items():
        if len(reps) == 1:
            profiles[next(iter(reps))].exclusive_obligations.add(key)

    for rep, profile in profiles.items():
        profile.mean_probe_words = sum(token_lists[rep]) / len(token_lists[rep]) if token_lists[rep] else math.inf
        total = 0.0
        for group in groups:
            if group.group_id not in profile.useful_group_ids:
                continue
            ranks = [m.rank for m in group.members if m.representation_id == rep]
            if ranks:
                total += 1.0 / (k0 + min(ranks))
        profile.rrf_contribution = total
    return profiles


def select_representations(profiles: dict[str, RepresentationProfile]) -> tuple[set[str], set[str]]:
    universe = set().union(*(p.obligations for p in profiles.values())) if profiles else set()
    if not universe:
        selected = set(profiles)
        for profile in profiles.values():
            profile.selected = True
            profile.selection_reasons.append("shared_probe_insufficiency")
        return selected, set()

    selected: set[str] = set()
    uncovered = set(universe)
    for profile in profiles.values():
        if profile.exclusive_obligations:
            selected.add(profile.representation_id)
            profile.selected = True
            profile.selection_reasons.append("exclusive_probe_obligation")
            uncovered -= profile.obligations

    while uncovered:
        candidates = [p for p in profiles.values() if p.representation_id not in selected]
        if not candidates:
            break
        candidates.sort(key=lambda p: (
            -len(p.obligations & uncovered),
            -p.rrf_contribution,
            p.mean_probe_words,
            p.representation_id,
        ))
        best = candidates[0]
        gain = best.obligations & uncovered
        if not gain:
            break
        selected.add(best.representation_id)
        best.selected = True
        best.selection_reasons.append("set_cover:" + ",".join(sorted(gain)))
        uncovered -= gain
    return selected, uncovered
