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

    @property
    def vector_nbytes(self) -> int:
        return int(self.vectors.nbytes)

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


class HierarchicalVectorIndex(ExactVectorIndex):
    """Two-level exact index for the versioned hierarchy candidate.

    Child and parent cosine scores are converted independently to empirical
    percentiles, then combined with equal weight.  Equal weighting is a neutral
    candidate setting, not a frozen claim; parent/child weighting must be
    assessed during version-2 training-only calibration.
    """

    def __init__(self, representation_id: str, chunks: list[Chunk], encoder: Encoder):
        super().__init__(representation_id, chunks, encoder)
        parent_text_by_id: dict[str, str] = {}
        for chunk in chunks:
            parent_id = str(chunk.metadata.get("hierarchy_parent_id", ""))
            parent_text = str(chunk.metadata.get("hierarchy_parent_embedding_text", ""))
            if not parent_id or not parent_text:
                raise ValueError("hierarchical_v2 chunks require parent metadata")
            existing = parent_text_by_id.setdefault(parent_id, parent_text)
            if existing != parent_text:
                raise ValueError(f"inconsistent parent text for {parent_id}")
        self.parent_ids = tuple(parent_text_by_id)
        prepared_parent_texts = [
            encoder.prepare_embedding_text(parent_text_by_id[parent_id], 256)[0]
            for parent_id in self.parent_ids
        ]
        self.parent_vectors = (
            encoder.encode(prepared_parent_texts)
            if prepared_parent_texts else np.empty((0, encoder.dimension), dtype=np.float32)
        )
        parent_position = {parent_id: position for position, parent_id in enumerate(self.parent_ids)}
        self._chunk_parent_positions = np.asarray([
            parent_position[str(chunk.metadata["hierarchy_parent_id"])] for chunk in chunks
        ], dtype=np.int64)

    @property
    def vector_nbytes(self) -> int:
        return int(self.vectors.nbytes + self.parent_vectors.nbytes)

    @staticmethod
    def _percentile_scores(values: np.ndarray) -> np.ndarray:
        """Stable empirical percentiles with average ranks for exact ties."""
        if not len(values):
            return np.empty(0, dtype=np.float32)
        ordering = np.argsort(values, kind="stable")
        output = np.empty(len(values), dtype=np.float32)
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and values[ordering[end]] == values[ordering[start]]:
                end += 1
            average_one_based_rank = ((start + 1) + end) / 2.0
            output[ordering[start:end]] = average_one_based_rank / len(values)
            start = end
        return output

    def _scores(self, query: str) -> np.ndarray:
        cached = self._query_scores.get(query)
        if cached is not None:
            return cached
        query_vector = self._query_vectors.get(query)
        if query_vector is None:
            query_vector = self.encoder.encode([query])[0]
            self._query_vectors[query] = query_vector
        child_cosine = self.vectors @ query_vector
        parent_cosine = self.parent_vectors @ query_vector
        child_percentile = self._percentile_scores(child_cosine)
        parent_percentile = self._percentile_scores(parent_cosine)
        scores = 0.5 * (
            child_percentile + parent_percentile[self._chunk_parent_positions]
        )
        self._query_scores[query] = scores
        return scores
