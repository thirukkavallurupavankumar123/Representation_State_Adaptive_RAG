from __future__ import annotations

from hashlib import sha256

import numpy as np

from .encoding import Encoder
from .models import Candidate, Chunk


class ExactVectorIndex:
    def __init__(self, representation_id: str, chunks: list[Chunk], encoder: Encoder):
        self.representation_id = representation_id
        self.chunks = chunks
        self.encoder = encoder
        self.vectors = encoder.encode([c.embedding_text for c in chunks]) if chunks else np.empty((0, encoder.dimension), dtype=np.float32)
        self._positions = {chunk.chunk_id: index for index, chunk in enumerate(chunks)}
        self._query_vectors: dict[str, np.ndarray] = {}
        self._query_scores: dict[str, np.ndarray] = {}

    def vector_for_chunk(self, chunk_id: str) -> np.ndarray:
        return self.vectors[self._positions[chunk_id]]

    def _scores(self, query: str) -> np.ndarray:
        cached = self._query_scores.get(query)
        if cached is not None:
            return cached
        query_vector = self._query_vectors.get(query)
        if query_vector is None:
            query_vector = self.encoder.encode([query])[0]
            self._query_vectors[query] = query_vector
        scores = self.vectors @ query_vector
        self._query_scores[query] = scores
        return scores

    def relevance_percentile(self, query: str, raw_score: float) -> float:
        scores = self._scores(query)
        return float(np.count_nonzero(scores <= raw_score) / len(scores)) if len(scores) else 0.0

    def score_scale(self, query: str) -> float:
        scores = self._scores(query)
        return max(float(np.std(scores)), 1e-6) if len(scores) else 1.0

    def search(self, query: str, start_rank: int, limit: int, stage: str) -> list[Candidate]:
        if start_rank < 1 or limit < 1:
            raise ValueError("rank and limit must be positive")
        if not self.chunks:
            return []
        scores = self._scores(query)
        ordering = np.argsort(-scores, kind="stable")
        selected = ordering[start_rank - 1:start_rank - 1 + limit]
        result = []
        for offset, chunk_index in enumerate(selected):
            rank = start_rank + offset
            chunk = self.chunks[int(chunk_index)]
            candidate_id = sha256(
                f"{stage}|{query}|{self.representation_id}|{chunk.chunk_id}|{rank}".encode()
            ).hexdigest()[:24]
            result.append(Candidate(
                candidate_id=candidate_id,
                representation_id=self.representation_id,
                chunk_id=chunk.chunk_id,
                rank=rank,
                raw_score=float(scores[int(chunk_index)]),
                retrieval_query=query,
                retrieval_stage=stage,
                canonical_unit_ids=chunk.canonical_unit_ids,
                document_id=chunk.document_id,
                version_id=chunk.version_id,
                word_count=chunk.word_count,
            ))
        return result
