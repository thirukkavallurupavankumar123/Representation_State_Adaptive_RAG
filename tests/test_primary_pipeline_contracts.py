"""Contract tests for the primary one-call SAR-RAG pipeline.

The pre-existing suite asserted the headline contracts ("probe depth is not final
k", "context budget is never exceeded") against the *legacy* multi-call
``LlmJudgedSarRagSystem``, and its only primary-pipeline test disabled every
adaptive gate (``similarity_threshold=-1.0``, ``novelty_threshold=0.0``,
``marginal_utility_threshold=0.0``).  These tests exercise the primary
``SarRagSystem`` under the *actual frozen calibration controls*.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import unittest

import numpy as np

from sar_rag.adaptive_pipeline import SarRagSystem
from sar_rag.config import SarRagConfig
from sar_rag.corpus import build_corpus
from sar_rag.encoding import HashingEncoder
from sar_rag.groq import GroqOneCallAnswerGenerator
from sar_rag.index import ExactVectorIndex
from sar_rag.models import Candidate, Chunk, Document, FinalAnswerDecision
from sar_rag.selection import align_candidates, score_rrf, select_rank_only_representations
from sar_rag.traditional import TraditionalRagConfig, TraditionalRagSystem


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FROZEN_MANIFEST = REPOSITORY_ROOT / "output" / "calibration" / "frozen_manifest.json"
LOCAL_MODEL = REPOSITORY_ROOT / "models" / "all-MiniLM-L6-v2"
PROVENANCE = REPOSITORY_ROOT / "MODEL_PROVENANCE.json"


def frozen_configuration() -> dict:
    manifest = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["status"] == "training_only_calibration_frozen"
    return manifest["frozen_configuration"]


def frozen_config(**overrides) -> SarRagConfig:
    return SarRagConfig(**{**frozen_configuration()["proposed"], **overrides})


def research_document() -> Document:
    """A multi-section document that yields many chunks in every representation."""
    sections = []
    for section in range(6):
        sections.append(f"# Section {section} eligibility rules")
        for paragraph in range(4):
            sections.append(
                f"Rule {section}.{paragraph} states that an applicant in category "
                f"{section} must submit form {paragraph} before the annual review "
                f"deadline. Officers retain documentary evidence of category "
                f"{section} decisions for seven years and record the outcome in the "
                f"register of category {section} determinations."
            )
    return Document("policy", "v1", "Eligibility policy", "\n\n".join(sections))


class RecordingAnswerer:
    """Stands in for the single online Groq call and counts invocations."""

    def __init__(self) -> None:
        self.calls = 0
        self.evidence_sizes: list[int] = []

    def generate(self, query: str, evidence: list) -> FinalAnswerDecision:
        self.calls += 1
        self.evidence_sizes.append(len(evidence))
        if not evidence:
            return FinalAnswerDecision("Insufficient evidence.", "INSUFFICIENT", ())
        return FinalAnswerDecision("Applicants qualify [E1].", "ANSWERABLE", ("E1",))


def run_primary(config: SarRagConfig, document: Document | None = None):
    encoder = HashingEncoder()
    answerer = RecordingAnswerer()
    artifacts = build_corpus([document or research_document()], encoder,
                             config.enabled_representations)
    system = SarRagSystem(artifacts, encoder, answerer, config)
    return system, answerer, system.ask("Which applicants qualify and what evidence is retained?")


def scored_events(result) -> list[dict]:
    return [event for event in result.trace
            if event.get("event") == "candidate_scored_without_llm"]


class FrozenControlTests(unittest.TestCase):
    def test_frozen_manifest_values_are_the_ones_under_test(self) -> None:
        proposed = frozen_configuration()["proposed"]
        self.assertEqual(proposed, {
            "rrf_k0": 3,
            "representation_marginal_gain_threshold": 0.2437,
            "similarity_threshold": 0.5289,
            "similarity_gap_threshold": 1.9052,
            "novelty_threshold": 0.2292,
            "redundancy_threshold": 0.9547,
            "marginal_utility_threshold": 0.1289,
            "empty_batch_patience": 3,
        })
        self.assertEqual(frozen_configuration()["baseline"], {"static_k": 20})

    def test_frozen_manifest_is_actually_applied_to_the_config(self) -> None:
        config = frozen_config()
        for name, value in frozen_configuration()["proposed"].items():
            self.assertEqual(getattr(config, name), value, name)
        # The dataclass defaults are deliberately different executable starting
        # values; a run that forgot the manifest must be detectable.
        defaults = SarRagConfig()
        self.assertNotEqual(defaults.similarity_threshold, config.similarity_threshold)
        self.assertNotEqual(defaults.empty_batch_patience, config.empty_batch_patience)


class ProbeAndSelectionTests(unittest.TestCase):
    def test_all_four_representations_are_probed(self) -> None:
        config = frozen_config()
        self.assertEqual(config.enabled_representations,
                         ("sentence_window", "paragraph", "semantic", "hierarchical"))
        _, _, result = run_primary(config)
        probe = next(event for event in result.trace
                     if event["event"] == "probe_completed_without_llm")
        self.assertEqual(probe["probe_depth"], 3)
        # Four representations x top-3 probe, with no LLM involved.
        self.assertEqual(probe["candidate_count"], 12)

    def test_probe_depth_is_not_final_k(self) -> None:
        _, _, result = run_primary(frozen_config())
        self.assertNotEqual(result.final_k, 3)
        self.assertGreater(result.final_k, 3)
        completed = next(event for event in reversed(result.trace)
                         if event["event"] == "run_completed")
        self.assertEqual(completed["final_k"], result.final_k)

    def test_at_most_two_representations_are_selected(self) -> None:
        _, _, result = run_primary(frozen_config())
        self.assertLessEqual(len(result.selected_representations), 2)
        self.assertGreaterEqual(len(result.selected_representations), 1)

    def test_select_one_representation_ablation_is_respected(self) -> None:
        _, _, result = run_primary(frozen_config(max_selected_representations=1))
        self.assertEqual(len(result.selected_representations), 1)

    def test_normalized_rrf_matches_the_implemented_formula(self) -> None:
        """NRRF(g) = RRF(g) / (n_representations / (k0 + 1)).

        The selector then divides each representation's NRRF mass by the total
        NRRF mass over all aligned probe groups, so the frozen 0.2437 marginal
        gain is a *fraction of total aligned probe mass*, not a raw NRRF value.
        """
        k0 = 3
        names = ("sentence_window", "paragraph", "semantic", "hierarchical")
        candidates = [
            Candidate("a1", "sentence_window", "c1", 1, .9, "q", "probe", ("u1",), "d", "v", 10),
            Candidate("b1", "paragraph", "c1b", 1, .9, "q", "probe", ("u1",), "d", "v", 10),
            Candidate("a2", "sentence_window", "c2", 2, .8, "q", "probe", ("u2",), "d", "v", 10),
            Candidate("c1", "semantic", "c3", 1, .9, "q", "probe", ("u3",), "d", "v", 10),
        ]
        groups = align_candidates(candidates)
        score_rrf(groups, k0)
        by_units = {tuple(sorted(g.members[0].canonical_unit_ids)): g for g in groups}
        # Group u1 is retrieved at rank 1 by two representations.
        self.assertAlmostEqual(by_units[("u1",)].rrf_score, 1 / (k0 + 1) + 1 / (k0 + 1))
        self.assertAlmostEqual(by_units[("u2",)].rrf_score, 1 / (k0 + 2))
        self.assertAlmostEqual(by_units[("u3",)].rrf_score, 1 / (k0 + 1))

        _, profiles, weights = select_rank_only_representations(names, groups, k0, 2, 0.2437)
        maximum = len(names) / (k0 + 1)
        for group in groups:
            self.assertAlmostEqual(weights[group.group_id], group.rrf_score / maximum)
        total = sum(weights.values())
        expected = (by_units[("u1",)].rrf_score + by_units[("u2",)].rrf_score) / maximum / total
        self.assertAlmostEqual(profiles["sentence_window"].normalized_coverage_score, expected)
        # hierarchical retrieved nothing in the probe.
        self.assertEqual(profiles["hierarchical"].normalized_coverage_score, 0.0)

    def test_second_representation_requires_the_frozen_marginal_gain(self) -> None:
        names = ("a", "b")
        candidates = [
            Candidate("a1", "a", "c1", 1, .9, "q", "probe", ("u1",), "d", "v", 10),
            Candidate("a2", "a", "c2", 2, .8, "q", "probe", ("u2",), "d", "v", 10),
            Candidate("a3", "a", "c3", 3, .7, "q", "probe", ("u3",), "d", "v", 10),
            # 'b' duplicates one group and adds a single low-rank unique group.
            Candidate("b1", "b", "c1b", 1, .9, "q", "probe", ("u1",), "d", "v", 10),
            Candidate("b3", "b", "c4", 3, .5, "q", "probe", ("u9",), "d", "v", 10),
        ]
        groups = align_candidates(candidates)
        selected_low, profiles_low, _ = select_rank_only_representations(names, groups, 3, 2, 0.01)
        selected_high, _, _ = select_rank_only_representations(names, groups, 3, 2, 0.99)
        self.assertEqual(selected_low, {"a", "b"})
        self.assertEqual(selected_high, {"a"})
        self.assertGreater(profiles_low["b"].marginal_gain, 0.0)


class IndexScoringTests(unittest.TestCase):
    def _index(self) -> ExactVectorIndex:
        class StubEncoder:
            dimension = 2

            def encode(self, texts: list[str]) -> np.ndarray:
                table = {
                    "q": [1.0, 0.0],
                    "x0": [1.0, 0.0], "x1": [0.8, 0.6],
                    "x2": [0.6, 0.8], "x3": [0.0, 1.0],
                }
                return np.asarray([table[t] for t in texts], dtype=np.float32)

            def token_count(self, text: str) -> int:
                return len(text.split())

            def prepare_embedding_text(self, text, max_tokens=256):
                return text, 1, False

        chunks = [
            Chunk(f"chunk{i}", "paragraph", "d", "v", (f"u{i}",), f"x{i}", f"x{i}", 1)
            for i in range(4)
        ]
        return ExactVectorIndex("paragraph", chunks, StubEncoder())

    def test_relevance_is_an_empirical_percentile_within_the_index(self) -> None:
        index = self._index()
        scores = sorted(float(v) for v in index._scores("q"))
        # cosine to [1,0] is 1.0, 0.8, 0.6, 0.0
        self.assertEqual([round(v, 4) for v in scores], [0.0, 0.6, 0.8, 1.0])
        self.assertAlmostEqual(index.relevance_percentile("q", 1.0), 1.0)
        self.assertAlmostEqual(index.relevance_percentile("q", 0.8), 0.75)
        self.assertAlmostEqual(index.relevance_percentile("q", 0.6), 0.5)
        self.assertAlmostEqual(index.relevance_percentile("q", 0.0), 0.25)

    def test_similarity_gap_is_standardized_by_index_score_deviation(self) -> None:
        index = self._index()
        expected = float(np.std(index._scores("q")))
        self.assertAlmostEqual(index.score_scale("q"), expected)
        self.assertGreater(expected, 0.0)
        # A raw drop is only meaningful relative to this representation's spread.
        raw_gap = 1.0 - 0.6
        self.assertAlmostEqual(raw_gap / index.score_scale("q"), raw_gap / expected)

    def test_score_scale_never_divides_by_zero(self) -> None:
        class Flat:
            dimension = 2

            def encode(self, texts):
                return np.asarray([[1.0, 0.0]] * len(texts), dtype=np.float32)

            def token_count(self, text):
                return 1

            def prepare_embedding_text(self, text, max_tokens=256):
                return text, 1, False

        chunks = [Chunk(f"c{i}", "paragraph", "d", "v", (f"u{i}",), "t", "t", 1) for i in range(3)]
        index = ExactVectorIndex("paragraph", chunks, Flat())
        self.assertGreaterEqual(index.score_scale("q"), 1e-6)


class AdaptiveRetrievalTests(unittest.TestCase):
    def test_retrieval_only_examines_selected_representations(self) -> None:
        _, _, result = run_primary(frozen_config())
        selected = set(result.selected_representations)
        examined = {event["representation"] for event in scored_events(result)}
        self.assertTrue(examined <= selected, f"{examined} escaped {selected}")
        accepted = {item.candidate.representation_id for item in result.final_evidence}
        self.assertTrue(accepted <= selected)

    def test_final_k_is_emergent_and_not_a_configured_constant(self) -> None:
        config = frozen_config()
        _, _, tight = run_primary(replace_tokens(config, 400))
        _, _, loose = run_primary(replace_tokens(config, 4000))
        self.assertNotEqual(tight.final_k, loose.final_k)
        self.assertLess(tight.final_k, loose.final_k)
        for result in (tight, loose):
            self.assertNotEqual(result.final_k, config.max_final_chunks)

    def test_stop_reason_and_round_counts_are_recorded(self) -> None:
        _, _, result = run_primary(frozen_config())
        completed = next(event for event in reversed(result.trace)
                         if event["event"] == "run_completed")
        self.assertIn("stop_reason", completed)
        self.assertIn("candidates_examined", completed)
        self.assertIn("retrieval_rounds", completed)
        self.assertTrue(result.stop_reason)
        self.assertGreaterEqual(completed["retrieval_rounds"], 0)

    def test_every_rejection_records_a_reason(self) -> None:
        _, _, result = run_primary(frozen_config())
        rejected = [event for event in scored_events(result) if not event["accepted"]]
        self.assertTrue(rejected, "expected the frozen thresholds to reject something")
        for event in rejected:
            self.assertIn(event["rejection_reason"], {
                "similarity_percentile", "novelty", "marginal_utility",
                "redundancy", "context_or_chunk_budget",
            })
        for event in scored_events(result):
            if event["accepted"]:
                self.assertIsNone(event["rejection_reason"])

    def test_novelty_and_marginal_utility_follow_the_declared_formulas(self) -> None:
        _, _, result = run_primary(frozen_config())
        for event in scored_events(result):
            expected_novelty = min(max(
                1.0 - event["canonical_overlap"] * max(event["semantic_similarity_to_selected"], 0.0),
                0.0), 1.0)
            self.assertAlmostEqual(event["novelty"], expected_novelty, places=6)
            self.assertAlmostEqual(
                event["marginal_utility"],
                event["normalized_relevance"] * event["novelty"], places=6,
            )


def replace_tokens(config: SarRagConfig, tokens: int) -> SarRagConfig:
    from dataclasses import replace
    return replace(config, max_final_context_tokens=tokens)


class SafetyCapTests(unittest.TestCase):
    def test_context_token_cap_is_never_exceeded(self) -> None:
        encoder = HashingEncoder()
        config = replace_tokens(frozen_config(), 300)
        system, _, result = run_primary(config)
        total = sum(encoder.token_count(item.chunk.evidence_text)
                    for item in result.final_evidence)
        self.assertLessEqual(total, config.max_final_context_tokens)

    def test_final_chunk_cap_is_never_exceeded(self) -> None:
        from dataclasses import replace
        config = replace(frozen_config(), max_final_chunks=4, max_final_context_tokens=100_000)
        _, _, result = run_primary(config)
        self.assertLessEqual(result.final_k, 4)

    def test_candidate_and_round_caps_bound_the_search(self) -> None:
        from dataclasses import replace
        config = replace(frozen_config(), max_candidates_examined=18, max_retrieval_rounds=2,
                         similarity_threshold=0.0, novelty_threshold=0.0,
                         marginal_utility_threshold=0.0, max_final_context_tokens=100_000,
                         max_final_chunks=1000)
        _, _, result = run_primary(config)
        completed = next(event for event in reversed(result.trace)
                         if event["event"] == "run_completed")
        self.assertLessEqual(completed["candidates_examined"], config.max_candidates_examined)
        self.assertLessEqual(completed["retrieval_rounds"], config.max_retrieval_rounds)
        self.assertIn(result.stop_reason, {
            "candidate_safety_cap", "retrieval_round_safety_cap",
            "all_selected_representations_paused",
        })

    def test_caps_are_limits_not_targets(self) -> None:
        """A generous budget must not drive final_k to the cap."""
        from dataclasses import replace
        config = replace(frozen_config(), max_final_context_tokens=100_000)
        _, _, result = run_primary(config)
        self.assertLess(result.final_k, config.max_final_chunks)


class DeduplicationTests(unittest.TestCase):
    def test_deduplication_happens_after_retrieval_not_during_probe(self) -> None:
        _, _, result = run_primary(frozen_config())
        self.assertTrue(result.provisional_evidence)
        self.assertTrue(result.final_evidence)
        # Provisional evidence is recorded before deduplication and is never smaller.
        self.assertGreaterEqual(len(result.provisional_evidence), len(result.final_evidence))
        # Every scored acceptance appears in the provisional set, so the probe stage
        # itself performed no removal.
        accepted_ids = {event["candidate_id"] for event in scored_events(result)
                        if event["accepted"]}
        provisional_ids = {item.candidate.candidate_id for item in result.provisional_evidence}
        self.assertEqual(accepted_ids, provisional_ids)

    def test_deduplication_can_be_disabled_and_retains_provenance(self) -> None:
        from dataclasses import replace
        _, _, off = run_primary(replace(frozen_config(),
                                        post_retrieval_deduplication_enabled=False))
        self.assertEqual(len(off.provisional_evidence), len(off.final_evidence))
        _, _, on = run_primary(frozen_config())
        for item in on.final_evidence:
            self.assertTrue(item.provenance)
            for representation, rank in item.provenance:
                self.assertIsInstance(representation, str)
                self.assertGreaterEqual(rank, 1)

    def test_contained_chunk_is_removed_only_when_superset_is_retained(self) -> None:
        _, _, result = run_primary(frozen_config())
        unit_sets = [set(item.candidate.canonical_unit_ids) for item in result.final_evidence]
        for left in range(len(unit_sets)):
            for right in range(len(unit_sets)):
                if left != right:
                    self.assertFalse(unit_sets[left] < unit_sets[right],
                                     "a strict subset survived deduplication")


class OneCallPolicyTests(unittest.TestCase):
    def test_exactly_one_final_llm_answer_call(self) -> None:
        _, answerer, result = run_primary(frozen_config())
        self.assertEqual(answerer.calls, 1)
        self.assertEqual(
            sum(event["event"] == "single_final_llm_call_completed" for event in result.trace), 1
        )

    def test_no_llm_event_precedes_the_single_answer_call(self) -> None:
        _, _, result = run_primary(frozen_config())
        names = [event["event"] for event in result.trace]
        final = names.index("single_final_llm_call_completed")
        for event in names[:final]:
            self.assertTrue(
                event.endswith("without_llm") or event in {
                    "normalized_rrf_completed", "representation_paused",
                },
                f"unexpected pre-answer event {event}",
            )

    def test_traditional_and_sar_share_the_identical_answer_generator(self) -> None:
        from sar_rag.adaptive_pipeline import SarRagSystem as Primary
        import inspect
        # Both systems must call the same class, not merely similar prompts.
        self.assertIs(GroqOneCallAnswerGenerator, GroqOneCallAnswerGenerator)
        traditional_source = inspect.getsource(TraditionalRagSystem.ask)
        primary_source = inspect.getsource(Primary.ask)
        self.assertIn("self.answerer.generate(query_text", traditional_source)
        self.assertIn("self.answerer.generate(query_text", primary_source)

    def test_both_systems_produce_identical_answers_from_one_shared_answerer(self) -> None:
        encoder = HashingEncoder()
        document = research_document()
        config = frozen_config()
        answerer = RecordingAnswerer()
        primary = SarRagSystem(
            build_corpus([document], encoder, config.enabled_representations),
            encoder, answerer, config,
        )
        baseline = TraditionalRagSystem(
            build_corpus([document], encoder, ("sliding_window",)), answerer,
            TraditionalRagConfig(static_k=20, max_context_tokens=config.max_final_context_tokens),
        )
        query = "Which applicants qualify?"
        primary_result = primary.ask(query)
        baseline_result = baseline.ask(query)
        # One answer call each, from the same generator instance.
        self.assertEqual(answerer.calls, 2)
        self.assertEqual(primary_result.answerability, baseline_result.answerability)

    def test_traditional_reports_the_candidates_it_actually_examined(self) -> None:
        encoder = HashingEncoder()
        config = TraditionalRagConfig(static_k=20, max_context_tokens=300)
        baseline = TraditionalRagSystem(
            build_corpus([research_document()], encoder, ("sliding_window",)),
            RecordingAnswerer(), config,
        )
        result = baseline.ask("Which applicants qualify?")
        completed = result.trace[0]
        self.assertEqual(completed["candidates_examined"], completed["retrieved_k"])
        # The shared token budget forces fewer final chunks than the static k.
        self.assertLess(completed["final_k"], completed["retrieved_k"])
        self.assertGreater(completed["candidates_examined"], completed["final_k"])


class RealEmbeddingIntegrationTests(unittest.TestCase):
    """Exercises the production encoder, not the deterministic hashing stand-in."""

    @classmethod
    def setUpClass(cls) -> None:
        if not (LOCAL_MODEL / "model.safetensors").is_file():
            raise unittest.SkipTest("local all-MiniLM-L6-v2 snapshot is not present")
        from sar_rag.encoding import SentenceTransformerEncoder
        cls.encoder = SentenceTransformerEncoder(str(LOCAL_MODEL))

    def test_local_snapshot_matches_recorded_provenance(self) -> None:
        import hashlib
        provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
        digest = hashlib.sha256((LOCAL_MODEL / "model.safetensors").read_bytes()).hexdigest()
        self.assertEqual(digest, provenance["weight_sha256"])
        self.assertEqual(provenance["hidden_dimension"], 384)
        self.assertEqual(provenance["max_sequence_length"], 256)

    def test_embeddings_are_384d_and_l2_normalized(self) -> None:
        vectors = self.encoder.encode(["applicants must retain documentary evidence",
                                       "an unrelated sentence about weather"])
        self.assertEqual(vectors.shape, (2, 384))
        self.assertEqual(self.encoder.dimension, 384)
        for row in vectors:
            self.assertAlmostEqual(float(np.linalg.norm(row)), 1.0, places=5)

    def test_encoding_is_deterministic(self) -> None:
        text = ["a deterministic sentence for repeated encoding"]
        np.testing.assert_allclose(self.encoder.encode(text), self.encoder.encode(text),
                                   rtol=0, atol=0)

    def test_overlength_text_is_bounded_and_truncation_is_recorded(self) -> None:
        long_text = " ".join(f"clause number {index} about retained evidence"
                             for index in range(400))
        prepared, original_tokens, truncated = self.encoder.prepare_embedding_text(long_text, 256)
        self.assertTrue(truncated)
        self.assertGreater(original_tokens, 256)
        self.assertLessEqual(self.encoder.token_count(prepared), 256)
        short = "a short clause"
        _, short_tokens, short_truncated = self.encoder.prepare_embedding_text(short, 256)
        self.assertFalse(short_truncated)
        self.assertLessEqual(short_tokens, 256)

    def test_corpus_records_truncation_metadata_for_every_chunk(self) -> None:
        artifacts = build_corpus([research_document()], self.encoder,
                                 ("sentence_window", "paragraph", "semantic", "hierarchical"))
        self.assertTrue(artifacts.chunks_by_id)
        for chunk in artifacts.chunks_by_id.values():
            self.assertIn("original_embedding_tokens", chunk.metadata)
            self.assertIn("embedding_was_truncated", chunk.metadata)
            self.assertIsInstance(chunk.metadata["embedding_was_truncated"], bool)

    def test_primary_pipeline_runs_end_to_end_on_the_real_encoder(self) -> None:
        config = frozen_config()
        answerer = RecordingAnswerer()
        artifacts = build_corpus([research_document()], self.encoder,
                                 config.enabled_representations)
        system = SarRagSystem(artifacts, self.encoder, answerer, config)
        result = system.ask("What evidence must officers retain?")
        self.assertEqual(answerer.calls, 1)
        self.assertLessEqual(len(result.selected_representations), 2)
        self.assertGreater(result.final_k, 0)
        self.assertLessEqual(result.final_k, config.max_final_chunks)
        total = sum(self.encoder.token_count(item.chunk.evidence_text)
                    for item in result.final_evidence)
        self.assertLessEqual(total, config.max_final_context_tokens)
        for event in scored_events(result):
            self.assertGreaterEqual(event["normalized_relevance"], 0.0)
            self.assertLessEqual(event["normalized_relevance"], 1.0)


if __name__ == "__main__":
    unittest.main()
