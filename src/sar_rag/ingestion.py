from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import re

from .models import CanonicalUnit, Document


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _digest(value: str, size: int = 24) -> str:
    return sha256(value.encode("utf-8")).hexdigest()[:size]


def parse_document(document: Document) -> list[CanonicalUnit]:
    """Parse Markdown/plain text into citable headings and sentence units."""
    units: list[CanonicalUnit] = []
    section_stack: list[tuple[int, str, str]] = []
    cursor = 0
    paragraph_id = 0
    blocks = re.split(r"(\n\s*\n)", document.text)

    for block in blocks:
        block_start = document.text.find(block, cursor)
        if block_start < 0:
            block_start = cursor
        cursor = block_start + len(block)
        text = block.strip()
        if not text:
            continue

        heading = HEADING_RE.match(text)
        if heading and "\n" not in text:
            level, title = len(heading.group(1)), heading.group(2).strip()
            while section_stack and section_stack[-1][0] >= level:
                section_stack.pop()
            unit_id = _digest(f"{document.version_id}|heading|{block_start}|{title}")
            section_stack.append((level, title, unit_id))
            units.append(CanonicalUnit(
                unit_id=unit_id,
                document_id=document.document_id,
                version_id=document.version_id,
                unit_type="heading",
                parent_id=section_stack[-2][2] if len(section_stack) > 1 else None,
                previous_unit_id=None,
                next_unit_id=None,
                section_path=tuple(x[1] for x in section_stack),
                character_start=block_start,
                character_end=block_start + len(block),
                raw_text=title,
                normalized_hash=_digest(" ".join(title.lower().split())),
                metadata={"heading_level": level, "paragraph_id": None},
            ))
            continue

        paragraph_id += 1
        parent_id = section_stack[-1][2] if section_stack else None
        section_path = tuple(x[1] for x in section_stack)
        local_offset = 0
        for sentence in SENTENCE_RE.split(text):
            sentence = sentence.strip()
            if not sentence:
                continue
            found = text.find(sentence, local_offset)
            found = local_offset if found < 0 else found
            start = block_start + found
            end = start + len(sentence)
            local_offset = found + len(sentence)
            unit_id = _digest(f"{document.version_id}|sentence|{start}|{sentence}")
            units.append(CanonicalUnit(
                unit_id=unit_id,
                document_id=document.document_id,
                version_id=document.version_id,
                unit_type="sentence",
                parent_id=parent_id,
                previous_unit_id=None,
                next_unit_id=None,
                section_path=section_path,
                character_start=start,
                character_end=end,
                raw_text=sentence,
                normalized_hash=_digest(" ".join(sentence.lower().split())),
                metadata={"paragraph_id": paragraph_id},
            ))

    linked: list[CanonicalUnit] = []
    for index, unit in enumerate(units):
        linked.append(replace(
            unit,
            previous_unit_id=units[index - 1].unit_id if index else None,
            next_unit_id=units[index + 1].unit_id if index + 1 < len(units) else None,
        ))
    validate_units(document, linked)
    return linked


def validate_units(document: Document, units: list[CanonicalUnit]) -> None:
    ids = {u.unit_id for u in units}
    if len(ids) != len(units):
        raise ValueError("duplicate canonical unit IDs")
    for unit in units:
        if unit.character_start < 0 or unit.character_end > len(document.text):
            raise ValueError(f"invalid offsets for {unit.unit_id}")
        if unit.parent_id is not None and unit.parent_id not in ids:
            raise ValueError(f"missing parent for {unit.unit_id}")
        if unit.previous_unit_id is not None and unit.previous_unit_id not in ids:
            raise ValueError(f"missing previous unit for {unit.unit_id}")
        if unit.next_unit_id is not None and unit.next_unit_id not in ids:
            raise ValueError(f"missing next unit for {unit.unit_id}")

