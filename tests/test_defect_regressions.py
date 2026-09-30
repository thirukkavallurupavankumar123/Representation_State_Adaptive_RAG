"""Regression tests for defects found during independent verification.

Each test fails against the pre-audit implementation and documents the defect it
pins, so the fix cannot silently regress.
"""

from __future__ import annotations

import math
import unittest

from sar_rag.evaluation import evidence_span_recall, macro_average
from sar_rag.experiments import _windows_peak_working_set_bytes
from sar_rag.models import Chunk, EvidenceItem, Judgment, PipelineResult, QueryAnalysis
from sar_rag.traditional import TraditionalRagConfig, TraditionalRagSystem
from sar_rag.corpus import build_corpus
from sar_rag.encoding import HashingEncoder
from sar_rag.models import Candidate, Document, FinalAnswerDecision


def _result(texts: list[str]) -> PipelineResult:
    evidence = []
    for index, text in enumerate(texts):
        candidate = Candidate(f"c{index}", "paragraph", f"chunk{index}", index + 1, 0.5,
                              "q", "adaptive", (f"u{index}",), "d", "v", len(text.split()))
        chunk = Chunk(f"chunk{index}", "paragraph", "d", "v", (f"u{index}",), text, text,
                      len(text.split()))
        evidence.append(EvidenceItem(candidate, chunk, Judgment(f"c{index}", ()), set()))
    return PipelineResult(
        query=QueryAnalysis("q", "query", ()), selected_representations=("paragraph",),
        representation_profiles={}, evidence_groups=[], provisional_evidence=evidence,
        final_evidence=evidence, final_k=len(evidence), answerability="ANSWERABLE",
        answer="answer", unresolved_requirements=(), stop_reason="x", trace=[],
        run_metadata={},
    )


class UnansweredRetrievalScoringTests(unittest.TestCase):
    """Defect: recall returned 1.0 when the benchmark declares no gold evidence.

    A question with no gold-evidence denominator scored a perfect retrieval
    result for retrieving nothing relevant, inflating any aggregate that mixed
    answerable and unanswerable questions.
    """

    def test_recall_is_undefined_without_gold_evidence(self) -> None:
        value = evidence_span_recall(_result(["irrelevant text"]), ())
        self.assertTrue(math.isnan(value), "no-gold recall must be undefined, not 1.0")

    def test_recall_is_still_computed_when_gold_evidence_exists(self) -> None:
        result = _result(["alpha beta gamma", "delta epsilon"])
        self.assertEqual(evidence_span_recall(result, ("alpha beta",)), 1.0)
        self.assertEqual(evidence_span_recall(result, ("alpha beta", "missing span")), 0.5)

    def test_macro_average_excludes_undefined_values_from_the_denominator(self) -> None:
        rows = [
            {"evidence_span_recall": 1.0, "final_k": 2.0},
            {"evidence_span_recall": float("nan"), "final_k": 4.0},
            {"evidence_span_recall": 0.0, "final_k": 6.0},
        ]
        averaged = macro_average(rows)
        # Mean over the two defined values, not over three with NaN propagated.
        self.assertEqual(averaged["evidence_span_recall"], 0.5)
        self.assertEqual(averaged["final_k"], 4.0)

    def test_macro_average_reports_nan_when_nothing_is_defined(self) -> None:
        averaged = macro_average([{"metric": float("nan")}, {"metric": float("nan")}])
        self.assertTrue(math.isnan(averaged["metric"]))


class TraditionalCandidateAccountingTests(unittest.TestCase):
    """Defect: the baseline reported final_k as candidates examined.

    method_metrics fell back to final_k because the baseline trace emitted only
    `retrieved_k`, understating the baseline's retrieval work whenever the shared
    token budget truncated the static top-k.
    """

    def test_baseline_trace_separates_examined_candidates_from_final_k(self) -> None:
        document = Document("d", "v1", "Policy", "\n\n".join(
            part for section in range(5)
            for part in ([f"# Section {section}"] + [
                f"Officers in category {section} retain evidence of decision {index} "
                f"for seven years and record the outcome in the register."
                for index in range(3)])
        ))

        class Answerer:
            def generate(self, query, evidence):
                return FinalAnswerDecision("a", "ANSWERABLE", ())

        encoder = HashingEncoder()
        system = TraditionalRagSystem(
            build_corpus([document], encoder, ("sliding_window",)), Answerer(),
            TraditionalRagConfig(static_k=20, max_context_tokens=120),
        )
        completed = system.ask("What must officers retain?").trace[0]
        self.assertIn("candidates_examined", completed)
        self.assertEqual(completed["candidates_examined"], completed["retrieved_k"])
        self.assertLess(completed["final_k"], completed["retrieved_k"],
                        "expected the token budget to truncate the static top-k")


class ProcessMemoryMeasurementTests(unittest.TestCase):
    """Defect: the 64-bit HANDLE was truncated, so the value was always null.

    GetCurrentProcess kept ctypes' default 32-bit int restype, so the
    (HANDLE)-1 pseudo-handle failed GetProcessMemoryInfo with
    ERROR_INVALID_HANDLE and the systems measurement recorded nothing.
    """

    def test_peak_working_set_is_reported_on_windows(self) -> None:
        import os
        value = _windows_peak_working_set_bytes()
        if os.name != "nt":
            self.assertIsNone(value)
            return
        self.assertIsNotNone(value, "peak working set must not be silently null")
        self.assertGreater(value, 1_000_000)


if __name__ == "__main__":
    unittest.main()
