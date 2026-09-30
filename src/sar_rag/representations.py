from __future__ import annotations

from collections import defaultdict
from hashlib import sha256

import numpy as np

from .encoding import Encoder
from .models import CanonicalUnit, Chunk, Document


def word_count(text: str) -> int:
    return len(text.split())


def _chunk_id(rep: str, version: str, unit_ids: list[str], suffix: str = "") -> str:
    value = f"{rep}|{version}|{'|'.join(unit_ids)}|{suffix}"
    return sha256(value.encode()).hexdigest()[:24]


def _make_chunk(rep: str, document: Document, units: list[CanonicalUnit], suffix: str = "", metadata=None) -> Chunk:
    section = " > ".join(units[0].section_path) if units and units[0].section_path else document.title
    evidence = "\n".join(u.raw_text for u in units)
    embedding = f"Document: {document.title}\nSection: {section}\n{evidence}"
    ids = [u.unit_id for u in units]
    return Chunk(
        chunk_id=_chunk_id(rep, document.version_id, ids, suffix),
        representation_id=rep,
        document_id=document.document_id,
        version_id=document.version_id,
        canonical_unit_ids=tuple(ids),
        embedding_text=embedding,
        evidence_text=evidence,
        word_count=word_count(evidence),
        metadata=metadata or {},
    )


def _bounded_groups(units: list[CanonicalUnit], max_words: int = 95) -> list[list[CanonicalUnit]]:
    groups, current, current_words = [], [], 0
    for unit in units:
        words = word_count(unit.raw_text)
        if current and current_words + words > max_words:
            groups.append(current)
            current, current_words = [], 0
        current.append(unit)
        current_words += words
    if current:
        groups.append(current)
    return groups


def sentence_windows(document: Document, units: list[CanonicalUnit], before=1, after=1) -> list[Chunk]:
    result = []
    text_units = [u for u in units if u.unit_type != "heading"]
    for index, target in enumerate(text_units):
        neighborhood = [
            u for u in text_units[max(0, index-before):index+after+1]
            if u.parent_id == target.parent_id
        ]
        same_parent = [target]
        for unit in sorted(
            (unit for unit in neighborhood if unit.unit_id != target.unit_id),
            key=lambda unit: abs(text_units.index(unit) - index),
        ):
            if sum(word_count(value.raw_text) for value in same_parent) + word_count(unit.raw_text) <= 95:
                same_parent.append(unit)
        same_parent.sort(key=lambda unit: unit.character_start)
        result.append(_make_chunk(
            "sentence_window", document, same_parent, suffix=target.unit_id,
            metadata={"target_unit_id": target.unit_id},
        ))
    return result


def sliding_windows(document: Document, units: list[CanonicalUnit], size=256, overlap=64,
                    representation: str = "sliding_window") -> list[Chunk]:
    source = [u for u in units if u.unit_type != "heading"]
    result: list[Chunk] = []
    start = 0
    while start < len(source):
        current: list[CanonicalUnit] = []
        count = 0
        end = start
        parent = source[start].parent_id
        while end < len(source):
            unit = source[end]
            if unit.parent_id != parent and current:
                break
            words = word_count(unit.raw_text)
            if current and count + words > size:
                break
            current.append(unit)
            count += words
            end += 1
        if current:
            result.append(_make_chunk(representation, document, current, suffix=str(start)))
        if end >= len(source):
            break
        retained = 0
        next_start = end
        while next_start > start + 1 and retained < overlap:
            next_start -= 1
            retained += word_count(source[next_start].raw_text)
        start = max(start + 1, next_start)
    return result


def paragraphs(document: Document, units: list[CanonicalUnit]) -> list[Chunk]:
    grouped: dict[tuple[str | None, int], list[CanonicalUnit]] = defaultdict(list)
    for unit in units:
        if unit.unit_type == "heading":
            continue
        grouped[(unit.parent_id, int(unit.metadata["paragraph_id"]))].append(unit)
    result = []
    for key, group in grouped.items():
        for part, bounded in enumerate(_bounded_groups(group)):
            result.append(_make_chunk("paragraph", document, bounded,
                                      suffix=f"{key[1]}:{part}"))
    return result


def semantic_chunks(document: Document, units: list[CanonicalUnit], encoder: Encoder,
                    min_words=32, max_words=95, boundary_percentile=80) -> list[Chunk]:
    source = [u for u in units if u.unit_type != "heading"]
    if not source:
        return []
    vectors = encoder.encode([u.raw_text for u in source])
    discontinuity = [1.0 - float(vectors[i] @ vectors[i+1]) for i in range(len(source)-1)]
    threshold = float(np.percentile(discontinuity, boundary_percentile)) if discontinuity else float("inf")
    result: list[Chunk] = []
    current: list[CanonicalUnit] = []
    current_words = 0
    for i, unit in enumerate(source):
        words = word_count(unit.raw_text)
        parent_changed = bool(current and current[-1].parent_id != unit.parent_id)
        exceeds = bool(current and current_words + words > max_words)
        semantic_break = bool(
            current and current_words >= min_words and i > 0 and discontinuity[i-1] >= threshold
        )
        if parent_changed or exceeds or semantic_break:
            result.append(_make_chunk("semantic", document, current, suffix=str(len(result))))
            current, current_words = [], 0
        current.append(unit)
        current_words += words
    if current:
        result.append(_make_chunk("semantic", document, current, suffix=str(len(result))))
    return result


def hierarchy(document: Document, units: list[CanonicalUnit]) -> list[Chunk]:
    """Legacy version-1 section chunks retained for exact reproducibility."""
    grouped: dict[str | None, list[CanonicalUnit]] = defaultdict(list)
    headings = {u.unit_id: u for u in units if u.unit_type == "heading"}
    for unit in units:
        if unit.unit_type != "heading":
            grouped[unit.parent_id].append(unit)
    result: list[Chunk] = []
    for parent_id, children in grouped.items():
        for part, bounded_children in enumerate(_bounded_groups(children)):
            content = ([headings[parent_id]] if parent_id in headings else []) + bounded_children
            if content:
                result.append(_make_chunk(
                    "hierarchical", document, content,
                    suffix=f"{parent_id or 'root'}:{part}",
                    metadata={"structural_parent_id": parent_id, "structural_part": part},
                ))
    return result


def hierarchy_v2(document: Document, units: list[CanonicalUnit]) -> list[Chunk]:
    """Versioned two-level hierarchy candidate with clean child evidence.

    Version 1 inserted the immediate heading into both ``evidence_text`` and an
    embedding string that already contained the section path.  On MuSiQue this
    repeated the title three times and behaved like a distorted paragraph view.

    This candidate keeps evidence children free of synthetic heading text and
    records one parent-section embedding for two-level rank fusion in
    ``HierarchicalVectorIndex``.  It is deliberately *not* substituted for the
    frozen ``hierarchical`` representation until fresh training-only screening
    and calibration support that change.
    """
    headings = {unit.unit_id: unit for unit in units if unit.unit_type == "heading"}
    grouped: dict[str | None, list[CanonicalUnit]] = defaultdict(list)
    for unit in units:
        if unit.unit_type != "heading":
            grouped[unit.parent_id].append(unit)

    result: list[Chunk] = []
    for parent_id, children in grouped.items():
        if not children:
            continue
        section_path = " > ".join(children[0].section_path) or document.title
        parent_heading = headings[parent_id].raw_text if parent_id in headings else document.title
        parent_key = _chunk_id(
            "hierarchical_v2_parent", document.version_id,
            [parent_id or document.document_id],
        )
        parent_embedding_text = (
            f"Document: {document.title}\n"
            f"Section path: {section_path}\n"
            f"Section heading: {parent_heading}\n"
            + "\n".join(child.raw_text for child in children)
        )
        for part, bounded_children in enumerate(_bounded_groups(children)):
            result.append(_make_chunk(
                "hierarchical_v2", document, bounded_children,
                suffix=f"{parent_id or 'root'}:{part}",
                metadata={
                    "hierarchy_parent_id": parent_key,
                    "hierarchy_parent_embedding_text": parent_embedding_text,
                    "structural_parent_id": parent_id,
                    "structural_part": part,
                    "hierarchy_version": 2,
                },
            ))
    return result


def build_all(document: Document, units: list[CanonicalUnit], encoder: Encoder,
              enabled: tuple[str, ...] | None = None) -> dict[str, list[Chunk]]:
    requested = set(enabled or (
        "sentence_window", "sliding_window", "paragraph", "semantic", "hierarchical"
    ))
    builders = {
        "sentence_window": lambda: sentence_windows(document, units),
        "sliding_window": lambda: sliding_windows(document, units),
        # Matched-granularity control.  The four adaptive views are bounded to ~95
        # words while the baseline sliding window is 256, so a context-token
        # difference between the methods confounds chunk size with adaptive
        # retrieval.  This variant holds granularity fixed at the adaptive views'
        # bound (overlap scaled by the same 0.25 ratio) so the two effects can be
        # separated.  It is a retrieval-only control, not a fifth primary view.
        "sliding_window_matched": lambda: sliding_windows(
            document, units, size=95, overlap=24,
            representation="sliding_window_matched",
        ),
        "paragraph": lambda: paragraphs(document, units),
        "semantic": lambda: semantic_chunks(document, units, encoder),
        "hierarchical": lambda: hierarchy(document, units),
        "hierarchical_v2": lambda: hierarchy_v2(document, units),
    }
    unknown = requested - builders.keys()
    if unknown:
        raise ValueError(f"unknown representations: {sorted(unknown)}")
    return {name: builders[name]() for name in builders if name in requested}
