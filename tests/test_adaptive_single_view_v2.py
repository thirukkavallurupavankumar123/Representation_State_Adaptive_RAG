from dataclasses import replace

import numpy as np
import pytest

from sar_rag.adaptive_single_view_v2 import (
    AdaptiveSingleViewV2Config,
    AdaptiveSingleViewV2System,
    canonical_sha256,
    validate_resume,
)
from sar_rag.corpus import CorpusArtifacts
from sar_rag.models import Candidate, Chunk


class FakeEncoder:
    dimension = 2

    def encode(self, texts):
        return np.ones((len(texts), 2), dtype=np.float32)

    def token_count(self, text):
        return len(text.split())

    def prepare_embedding_text(self, text, max_tokens=256):
        return text, self.token_count(text), False


class FakeIndex:
    representation_id = "semantic"

    def __init__(self, chunks, scores=None):
        self.chunks = chunks
        self.scores = scores or [1 - i / 100 for i in range(len(chunks))]

    @property
    def vector_nbytes(self):
        return 0

    def relevance_percentile(self, query, raw_score):
        return raw_score

    def search(self, query, start_rank, limit, stage):
        output = []
        for index in range(start_rank - 1, min(start_rank - 1 + limit, len(self.chunks))):
            chunk = self.chunks[index]
            output.append(Candidate(
                candidate_id=f"{stage}-{query}-{index}", representation_id="semantic",
                chunk_id=chunk.chunk_id, rank=index + 1, raw_score=self.scores[index],
                retrieval_query=query, retrieval_stage=stage,
                canonical_unit_ids=chunk.canonical_unit_ids,
                document_id=chunk.document_id, version_id=chunk.version_id,
                word_count=chunk.word_count,
            ))
        return output

    def vector_for_chunk(self, chunk_id):
        return np.asarray([1.0, 0.0], dtype=np.float32)


def chunks(count=12, words=2, duplicate_second=False):
    output = []
    for index in range(count):
        unit = "u0" if duplicate_second and index == 1 else f"u{index}"
        text = " ".join([f"w{index}"] * words)
        output.append(Chunk(
            chunk_id=f"c{index}", representation_id="semantic", document_id="d",
            version_id="v", canonical_unit_ids=(unit,), embedding_text=text,
            evidence_text=text, word_count=words,
        ))
    return output


def system(config=None, *, scores=None, chunk_values=None):
    config = config or AdaptiveSingleViewV2Config("semantic", 800)
    chunk_values = chunk_values or chunks()
    index = FakeIndex(chunk_values, scores)
    artifacts = CorpusArtifacts({}, {chunk.chunk_id: chunk for chunk in chunk_values}, {"semantic": index})
    return AdaptiveSingleViewV2System(artifacts, FakeEncoder(), config, "embedding-test")


def test_single_view_enforced_in_config_and_corpus():
    with pytest.raises(ValueError, match="exactly one"):
        AdaptiveSingleViewV2Config("semantic,paragraph", 800)
    config = AdaptiveSingleViewV2Config("semantic", 800)
    artifact = CorpusArtifacts({}, {}, {"semantic": FakeIndex([]), "paragraph": FakeIndex([])})
    with pytest.raises(ValueError, match="exactly"):
        AdaptiveSingleViewV2System(artifact, FakeEncoder(), config, "fp")


def test_adaptive_retrieval_continues_beyond_probe_and_final_k_is_emergent():
    result = system().retrieve("q")
    assert result.candidates_examined > 3
    assert result.retrieval_rounds > 0
    assert result.final_k == 12
    assert result.final_k != result.thresholds["probe_depth"]


def test_patience_stopping():
    config = AdaptiveSingleViewV2Config("semantic", 800, relevance_percentile_threshold=0.8,
                                       empty_batch_patience=2)
    result = system(config, scores=[0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]).retrieve("q")
    assert result.stop_reason == "patience_no_gain"
    assert result.retrieval_rounds == 2


@pytest.mark.parametrize(
    ("config", "chunk_values", "expected"),
    [
        (AdaptiveSingleViewV2Config("semantic", 800, max_candidates_examined=4), chunks(12),
         "candidate_safety_cap"),
        (AdaptiveSingleViewV2Config("semantic", 800, max_retrieval_rounds=1), chunks(12),
         "retrieval_round_safety_cap"),
        (AdaptiveSingleViewV2Config("semantic", 800, max_final_chunks=2), chunks(12),
         "final_chunk_safety_cap"),
        (AdaptiveSingleViewV2Config("semantic", 800), chunks(4, words=500),
         "context_token_safety_cap"),
    ],
)
def test_each_safety_cap(config, chunk_values, expected):
    assert system(config, chunk_values=chunk_values).retrieve("q").stop_reason == expected


def test_token_accounting_and_dedup_happen_after_retrieval():
    result = system(chunk_values=chunks(5, words=7, duplicate_second=True)).retrieve("q")
    assert result.pre_dedup_k == 5
    assert result.post_dedup_k == result.final_k == 4
    assert result.context_tokens == 28
    events = [event["event"] for event in result.trace]
    assert events.index("v2_post_retrieval_deduplication_completed") > max(
        index for index, event in enumerate(events) if event == "v2_candidate_scored_without_llm")


def test_probe_only_is_explicit_comparator_not_silent_fixed_k():
    config = AdaptiveSingleViewV2Config("semantic", 800, adaptive_retrieval_enabled=False)
    result = system(config).retrieve("q")
    assert result.stop_reason == "probe_only_comparator"
    assert result.final_k == 3
    assert result.candidates_examined == 3


def test_resume_rejects_every_fingerprint_mismatch():
    expected = {name: name for name in (
        "method", "configuration_fingerprint", "embedding_fingerprint", "input_sha256",
        "protocol_sha256", "code_sha256")}
    validate_resume(dict(expected), expected)
    for field in expected:
        existing = dict(expected)
        existing[field] = "different"
        with pytest.raises(ValueError, match=field):
            validate_resume(existing, expected)


def test_retrieval_has_no_answerer_or_groq_call_and_is_deterministic():
    retriever = system()
    assert not hasattr(retriever, "answerer")
    first = retriever.retrieve("same")
    second = retriever.retrieve("same")
    assert first.configuration_fingerprint == canonical_sha256(retriever.config.__dict__)
    assert [item.chunk.chunk_id for item in first.final_evidence] == [
        item.chunk.chunk_id for item in second.final_evidence]
    assert first.stop_reason == second.stop_reason
    assert all(event.get("llm_calls", 0) == 0 for event in first.trace)
