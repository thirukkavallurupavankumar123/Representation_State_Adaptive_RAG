from __future__ import annotations

import unittest
import json
import math
from sar_rag.calibration import (
    choose_configuration, deterministic_stratified_sample, diverse_smoke_sample,
)

from sar_rag.config import SarRagConfig
from sar_rag.ablations import apply_condition
from sar_rag.corpus import build_corpus
from sar_rag.encoding import HashingEncoder
from sar_rag.evidence import deduplicate_after_retrieval, missing_applicability
from sar_rag.evaluation import exact_match, token_f1, token_prf
from sar_rag.experiments import exact_mcnemar_p_value, holm_adjust, retrieval_metrics
from sar_rag.q2_analysis import descriptive
from sar_rag.groq import (
    GroqAnswerGenerator, GroqEvidenceJudge, GroqOneCallAnswerGenerator,
    GroqRequirementExtractor,
)
from sar_rag.index import HierarchicalVectorIndex
from sar_rag.models import (
    Candidate, Chunk, Document, EvidenceItem, Finding, Judgment, QueryAnalysis,
    FinalAnswerDecision, Requirement,
)
from sar_rag.pipeline import LlmJudgedSarRagSystem, SarRagSystem
from sar_rag.reasoning import (
    EvidenceOnlyAnswerGenerator, ScriptedJudge, ScriptedRequirementExtractor,
    SimpleReformulator,
)
from sar_rag.selection import align_candidates, score_rrf, select_rank_only_representations


def _candidate(candidate_id: str, representation: str, rank: int,
               unit_ids: tuple[str, ...]) -> Candidate:
    return Candidate(
        candidate_id, representation, candidate_id + "-chunk", rank, 0.9, "query", "probe",
        unit_ids, "doc", "v1", 10,
    )


class SelectionTests(unittest.TestCase):
    def test_rrf_uses_one_best_rank_per_representation(self) -> None:
        candidates = [
            _candidate("a", "sentence_window", 1, ("u1",)),
            _candidate("b", "sentence_window", 2, ("u1",)),
            _candidate("c", "paragraph", 3, ("u1", "u2")),
        ]
        groups = align_candidates(candidates)
        self.assertEqual(len(groups), 1)
        score_rrf(groups, 60)
        self.assertAlmostEqual(groups[0].rrf_score, 1 / 61 + 1 / 63)

    def test_rank_only_selector_adds_only_complementary_second_view(self) -> None:
        candidates = [
            _candidate("a1", "a", 1, ("u1",)),
            _candidate("a2", "a", 2, ("u2",)),
            _candidate("b1", "b", 1, ("u1",)),
            _candidate("b2", "b", 2, ("u3",)),
            _candidate("c1", "c", 1, ("u1",)),
            _candidate("c2", "c", 2, ("u2",)),
        ]
        groups = align_candidates(candidates)
        selected, profiles, _ = select_rank_only_representations(
            ("a", "b", "c"), groups, k0=3, max_selected=2,
            marginal_gain_threshold=0.01,
        )
        self.assertEqual(selected, {"a", "b"})
        self.assertGreater(profiles["b"].marginal_gain, 0)
        self.assertFalse(profiles["c"].selected)


class EvidenceTests(unittest.TestCase):
    def test_post_retrieval_exact_dedup_preserves_provenance(self) -> None:
        unit_ids = ("u1",)
        chunk_a = Chunk("ca", "paragraph", "doc", "v1", unit_ids, "x", "x", 1)
        chunk_b = Chunk("cb", "semantic", "doc", "v1", unit_ids, "x", "x", 1)
        finding = Finding("R1", "SUPPORTED", "x", unit_ids)
        item_a = EvidenceItem(
            _candidate("a", "paragraph", 1, unit_ids), chunk_a,
            Judgment("a", (finding,)), {"REQ:R1:SUPPORTED"}, [("paragraph", 1)],
        )
        item_b = EvidenceItem(
            _candidate("b", "semantic", 2, unit_ids), chunk_b,
            Judgment("b", (finding,)), {"REQ:R1:SUPPORTED"}, [("semantic", 2)],
        )
        result = deduplicate_after_retrieval([item_a, item_b], {"REQ:R1:SUPPORTED"})
        self.assertEqual(len(result), 1)
        self.assertEqual(set(result[0].provenance), {("paragraph", 1), ("semantic", 2)})

    def test_requested_applicability_must_have_a_value(self) -> None:
        requirement = Requirement("R1", "Who qualifies?", applicability_variables=("age", "region"))
        candidate = _candidate("a", "paragraph", 1, ("u1",))
        chunk = Chunk("ca", "paragraph", "doc", "v1", ("u1",), "x", "x", 1)
        finding = Finding("R1", "SUPPORTED", "x", ("u1",), (("age", ("under 18",)),))
        item = EvidenceItem(candidate, chunk, Judgment("a", (finding,)), finding.obligation_keys())
        missing = missing_applicability((requirement,), [item])
        self.assertEqual(missing["R1"], {"region"})


class PipelineTests(unittest.TestCase):
    def _system(self, include_scope: bool, token_budget: int = 2000) -> LlmJudgedSarRagSystem:
        query = "Who qualifies and what age rule applies?"
        analysis = QueryAnalysis(
            "q1", query,
            (Requirement("R1", "Identify who qualifies and the age rule.", True,
                         applicability_variables=("age",)),),
        )

        def resolver(parsed_query, candidate, chunk):
            applicable = (("age", ("under 18",)),) if include_scope else ()
            finding = Finding(
                "R1", "SUPPORTED", "Applicants under 18 qualify only with guardian consent.",
                candidate.canonical_unit_ids, applicable,
            )
            return Judgment(candidate.candidate_id, (finding,), valid=True)

        document = Document(
            "policy", "v1", "Eligibility policy",
            "# Eligibility\n\nApplicants under 18 qualify only with guardian consent.",
        )
        config = SarRagConfig(
            probe_depth=3,
            adaptive_batch_size=2,
            max_candidates_examined=30,
            max_retrieval_rounds=6,
            max_final_context_tokens=token_budget,
        )
        return LlmJudgedSarRagSystem.with_components(
            [document], HashingEncoder(), ScriptedRequirementExtractor(analysis),
            ScriptedJudge(resolver), SimpleReformulator(), EvidenceOnlyAnswerGenerator(), config,
        )

    def test_probe_depth_is_not_final_k_and_dedup_is_after_retrieval(self) -> None:
        result = self._system(include_scope=True).ask("Who qualifies and what age rule applies?")
        self.assertEqual(result.answerability, "ANSWERABLE")
        self.assertEqual(result.final_k, 1)
        self.assertNotEqual(result.final_k, 3)
        completed = next(event for event in result.trace if event["event"] == "probe_completed")
        self.assertEqual(completed["probe_depth"], 3)
        self.assertEqual(result.stop_reason, "evidence_sufficient_after_empty_confirmation")

    def test_missing_applicability_prevents_complete_answer(self) -> None:
        result = self._system(include_scope=False).ask("Who qualifies and what age rule applies?")
        self.assertIn("R1", result.unresolved_requirements)
        self.assertNotEqual(result.answerability, "ANSWERABLE")

    def test_context_budget_is_never_silently_exceeded(self) -> None:
        result = self._system(include_scope=True, token_budget=1).ask(
            "Who qualifies and what age rule applies?"
        )
        self.assertEqual(result.answerability, "INSUFFICIENT_UNDER_BUDGET")
        self.assertLessEqual(sum(item.chunk.word_count for item in result.final_evidence), 1)
        self.assertTrue(any(event["event"] == "context_budget_insufficient" for event in result.trace))


class _OneCallAnswerer:
    def __init__(self):
        self.calls = 0

    def generate(self, query, evidence):
        self.calls += 1
        if not evidence:
            return FinalAnswerDecision("The retrieved evidence is insufficient.", "INSUFFICIENT")
        return FinalAnswerDecision("Applicants qualify [E1].", "ANSWERABLE", ("E1",))


class OneCallPipelineTests(unittest.TestCase):
    def test_primary_pipeline_uses_one_answer_call_and_at_most_two_views(self) -> None:
        document = Document(
            "policy", "v1", "Eligibility",
            "# Eligibility\n\nApplicants qualify when their registration is active.\n\n"
            "# Conditions\n\nRegistration must be renewed every year.",
        )
        encoder = HashingEncoder()
        config = SarRagConfig(
            similarity_threshold=-1.0,
            novelty_threshold=0.0,
            marginal_utility_threshold=0.0,
            max_retrieval_rounds=2,
            max_candidates_examined=20,
            max_final_context_tokens=500,
        )
        answerer = _OneCallAnswerer()
        system = SarRagSystem(
            build_corpus([document], encoder, config.enabled_representations),
            encoder, answerer, config,
        )
        result = system.ask("Who qualifies?")
        self.assertEqual(answerer.calls, 1)
        self.assertLessEqual(len(result.selected_representations), 2)
        self.assertGreater(result.final_k, 0)
        self.assertEqual(
            sum(event["event"] == "single_final_llm_call_completed" for event in result.trace), 1
        )
        self.assertTrue(any(event["event"] == "normalized_rrf_completed" for event in result.trace))

    def test_long_sections_are_partitioned_before_hierarchical_indexing(self) -> None:
        paragraphs = "\n\n".join(
            "Sentence number %d contains enough repeated policy language for bounded grouping. "
            "Applicants must retain documentary evidence for review." % index
            for index in range(80)
        )
        document = Document("long", "v1", "Long policy", "# Rules\n\n" + paragraphs)
        encoder = HashingEncoder()
        artifacts = build_corpus([document], encoder, ("hierarchical",))
        chunks = artifacts.indexes["hierarchical"].chunks
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.metadata["original_embedding_tokens"] < 300 for chunk in chunks))

    def test_hierarchical_v2_is_versioned_and_keeps_headings_out_of_evidence(self) -> None:
        document = Document(
            "policy", "v1", "Policy title",
            "# Eligibility\n\nApplicants qualify when registration is active. "
            "Evidence must be retained.\n\nRenewal happens annually.",
        )
        artifacts = build_corpus([document], HashingEncoder(), ("hierarchical_v2",))
        index = artifacts.indexes["hierarchical_v2"]
        self.assertIsInstance(index, HierarchicalVectorIndex)
        self.assertGreater(index.vector_nbytes, index.vectors.nbytes)
        self.assertTrue(index.chunks)
        for chunk in index.chunks:
            self.assertEqual(chunk.metadata["hierarchy_version"], 2)
            self.assertNotIn("Eligibility\n", chunk.evidence_text)
            self.assertIn("Applicants qualify", chunk.evidence_text)
            self.assertIn("Section heading: Eligibility", chunk.metadata["hierarchy_parent_embedding_text"])


class _FakeGroqClient:
    def __init__(self, values):
        self.values = iter(values)

    def complete(self, *args, **kwargs):
        return next(self.values)


class GroqContractTests(unittest.TestCase):
    def test_requirement_extractor_accepts_only_structured_output(self) -> None:
        client = _FakeGroqClient([json.dumps({
            "requirements": [{
                "text": "Determine eligibility", "critical": True,
                "entities": ["applicant"], "applicability_variables": ["age"],
            }],
            "query_facts": [], "decomposition_confident": True,
        })])
        analysis = GroqRequirementExtractor(client).analyze("Who qualifies?")
        self.assertEqual(analysis.requirements[0].applicability_variables, ("age",))

    def test_judge_rejects_non_cited_supported_finding(self) -> None:
        client = _FakeGroqClient([json.dumps({
            "findings": [{
                "requirement_id": "R1", "status": "SUPPORTED", "claim": "claim",
                "cited_unit_ids": ["invented-unit"], "applicability": [], "qualifiers": [],
            }],
            "context_keys": [], "contradiction_keys": [],
        })])
        analysis = QueryAnalysis("q", "query", (Requirement("R1", "requirement"),))
        candidate = _candidate("a", "paragraph", 1, ("u1",))
        chunk = Chunk("ca", "paragraph", "doc", "v1", ("u1",), "x", "x", 1)
        judgment = GroqEvidenceJudge(client).judge(analysis, candidate, chunk)
        self.assertFalse(judgment.valid)

    def test_answer_generator_rejects_invented_evidence_number(self) -> None:
        client = _FakeGroqClient([json.dumps({
            "answer": "Claim [E2]", "answerability": "ANSWERABLE",
            "unresolved_requirements": [],
        })])
        analysis = QueryAnalysis("q", "query", (Requirement("R1", "requirement"),))
        candidate = _candidate("a", "paragraph", 1, ("u1",))
        chunk = Chunk("ca", "paragraph", "doc", "v1", ("u1",), "x", "x", 1)
        finding = Finding("R1", "SUPPORTED", "x", ("u1",))
        evidence = [EvidenceItem(candidate, chunk, Judgment("a", (finding,)), finding.obligation_keys())]
        with self.assertRaises(RuntimeError):
            GroqAnswerGenerator(client).generate(analysis, evidence, (), "ANSWERABLE")

    def test_one_call_generator_derives_valid_inline_citations(self) -> None:
        client = _FakeGroqClient([json.dumps({
            "answerability": "ANSWERABLE",
            "claims": [{"text": "The evidence supports the result.", "evidence_ids": ["E1"]}],
            "missing_information": "",
        })])
        candidate = _candidate("a", "paragraph", 1, ("u1",))
        chunk = Chunk("ca", "paragraph", "doc", "v1", ("u1",), "x", "x", 1)
        evidence = [EvidenceItem(candidate, chunk, Judgment("a", ()), set())]
        decision = GroqOneCallAnswerGenerator(client).generate("What is supported?", evidence)
        self.assertEqual(decision.cited_evidence_ids, ("E1",))
        self.assertIn("[E1]", decision.answer)


class EvaluationTests(unittest.TestCase):
    def test_answer_metrics_ignore_evidence_citation_markers(self) -> None:
        self.assertEqual(exact_match("The policy applies. [E1]", ("policy applies",)), 1.0)
        self.assertEqual(token_f1("The policy applies [E1]", ("policy applies",)), 1.0)

    def test_retrieval_metrics_are_undefined_without_gold_evidence(self) -> None:
        metrics = retrieval_metrics([], ())
        self.assertTrue(all(math.isnan(value) for value in metrics.values()))

    def test_token_prf_uses_gold_with_best_f1(self) -> None:
        precision, recall, f1 = token_prf("alpha beta", ("wrong", "alpha beta gamma"))
        self.assertEqual(precision, 1.0)
        self.assertAlmostEqual(recall, 2 / 3)
        self.assertAlmostEqual(f1, 0.8)

    def test_complete_evidence_requires_all_unique_gold_spans(self) -> None:
        candidate = _candidate("a", "paragraph", 1, ("u1",))
        chunk = Chunk("a-chunk", "paragraph", "doc", "v1", ("u1",), "alpha", "alpha", 1)
        item = EvidenceItem(candidate, chunk, Judgment("a", ()), set())
        metrics = retrieval_metrics([item], ("alpha", "beta"))
        self.assertEqual(metrics["evidence_set_recall"], 0.5)
        self.assertEqual(metrics["complete_evidence_recovery"], 0.0)

    def test_exact_mcnemar_and_holm_are_deterministic(self) -> None:
        result = exact_mcnemar_p_value([1, 1, 0, 0], [1, 0, 1, 1])
        self.assertEqual(result["discordant_pairs"], 3)
        adjusted = holm_adjust({"a": .01, "b": .04, "c": .20})
        self.assertAlmostEqual(adjusted["a"], .03)
        self.assertAlmostEqual(adjusted["b"], .08)

    def test_descriptive_statistics_include_tail_latency(self) -> None:
        result = descriptive([1, 2, 3, 4, 100])
        self.assertEqual(result["n"], 5)
        self.assertGreater(result["p95"], result["median"])

    def test_registered_ablation_changes_only_declared_control(self) -> None:
        config, static_k, engine = apply_condition(SarRagConfig(), 20, "no_novelty")
        self.assertFalse(config.novelty_enabled)
        self.assertEqual(static_k, 20)
        self.assertEqual(engine, "one_call")


class CalibrationTests(unittest.TestCase):
    def test_stratified_sample_is_order_independent(self) -> None:
        rows = [{
            "example_id": f"qasper:train:{index}", "dataset": "qasper", "split": "train",
            "gold_answerability": "ANSWERABLE", "documents": [{"text": "x " * 10}],
        } for index in range(8)]
        forward, _ = deterministic_stratified_sample(rows, 3, 42)
        backward, _ = deterministic_stratified_sample(reversed(rows), 3, 42)
        self.assertEqual([row["example_id"] for row in forward],
                         [row["example_id"] for row in backward])

    def test_calibration_prefers_efficiency_only_within_recall_tolerance(self) -> None:
        results = [
            {"parameters": {"name": "best"}, "metrics": {
                "evidence_recall": 1.0, "complete_evidence": .8,
                "evidence_precision": .4, "context_tokens": 1000, "candidates_examined": 20,
            }},
            {"parameters": {"name": "efficient"}, "metrics": {
                "evidence_recall": .995, "complete_evidence": .8,
                "evidence_precision": .4, "context_tokens": 500, "candidates_examined": 10,
            }},
            {"parameters": {"name": "too_low"}, "metrics": {
                "evidence_recall": .98, "complete_evidence": 1.0,
                "evidence_precision": 1.0, "context_tokens": 10, "candidates_examined": 1,
            }},
        ]
        self.assertEqual(choose_configuration(results)["parameters"]["name"], "efficient")

    def test_smoke_sample_spans_dataset_before_repeating(self) -> None:
        rows = [
            {"dataset": "qasper", "gold_answerability": "ANSWERABLE", "example_id": "q1"},
            {"dataset": "qasper", "gold_answerability": "INSUFFICIENT", "example_id": "q2"},
            {"dataset": "musique_full", "gold_answerability": "ANSWERABLE", "example_id": "m1"},
            {"dataset": "musique_full", "gold_answerability": "INSUFFICIENT", "example_id": "m2"},
        ]
        sample = diverse_smoke_sample(rows, 4)
        self.assertEqual(len({(row["dataset"], row["gold_answerability"]) for row in sample}), 4)


if __name__ == "__main__":
    unittest.main()
