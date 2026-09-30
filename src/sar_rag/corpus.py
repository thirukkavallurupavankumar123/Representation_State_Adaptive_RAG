from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from .encoding import Encoder
from .index import ExactVectorIndex, HierarchicalVectorIndex
from .ingestion import parse_document
from .models import Chunk, Document
from .representations import build_all


@dataclass
class CorpusArtifacts:
    units_by_id: dict
    chunks_by_id: dict[str, Chunk]
    indexes: dict[str, ExactVectorIndex]

    @property
    def index_vector_bytes(self) -> int:
        return sum(index.vector_nbytes for index in self.indexes.values())

    @property
    def indexed_chunks(self) -> int:
        return sum(len(index.chunks) for index in self.indexes.values())


def build_corpus(documents: Iterable[Document], encoder: Encoder,
                 enabled: tuple[str, ...]) -> CorpusArtifacts:
    by_rep: dict[str, list[Chunk]] = {rep: [] for rep in enabled}
    all_units = {}
    for document in documents:
        units = parse_document(document)
        all_units.update({unit.unit_id: unit for unit in units})
        representations = build_all(document, units, encoder, enabled)
        for rep in enabled:
            for chunk in representations[rep]:
                embedding_text, original_embedding_tokens, was_truncated = (
                    encoder.prepare_embedding_text(chunk.embedding_text, 256)
                )
                by_rep[rep].append(replace(chunk, metadata={
                    **chunk.metadata,
                    "document_title": document.title,
                    "effective_date": document.effective_date,
                    "authority_level": document.authority_level,
                    "supersedes_version_id": document.supersedes_version_id,
                    "original_embedding_tokens": original_embedding_tokens,
                    "embedding_was_truncated": was_truncated,
                }, embedding_text=embedding_text))
    chunks = {chunk.chunk_id: chunk for values in by_rep.values() for chunk in values}
    indexes = {
        rep: (
            HierarchicalVectorIndex(rep, values, encoder)
            if rep == "hierarchical_v2" else ExactVectorIndex(rep, values, encoder)
        )
        for rep, values in by_rep.items()
    }
    return CorpusArtifacts(all_units, chunks, indexes)
