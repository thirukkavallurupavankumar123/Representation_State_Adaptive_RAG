from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("screening_v2", ROOT / "tools" / "screening_v2.py")
assert SPEC and SPEC.loader
SCREENING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCREENING)


class ScreeningV2AnalysisTests(unittest.TestCase):
    def test_holm_adjust_is_monotone_in_sorted_p_order(self) -> None:
        adjusted = SCREENING.holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["c"], 0.06)
        self.assertAlmostEqual(adjusted["b"], 0.06)

    def test_all_six_legacy_representation_pairs_are_registered(self) -> None:
        self.assertEqual(len(SCREENING.REPRESENTATION_PAIRS), 6)
        self.assertEqual(len(set(SCREENING.REPRESENTATION_PAIRS)), 6)
        for left, right in SCREENING.REPRESENTATION_PAIRS:
            self.assertNotEqual(left, right)
            self.assertIn(left, SCREENING.SINGLE_VIEWS)
            self.assertIn(right, SCREENING.SINGLE_VIEWS)

    def test_all_six_candidate_representation_pairs_are_registered(self) -> None:
        self.assertEqual(len(SCREENING.CANDIDATE_REPRESENTATION_PAIRS), 6)
        self.assertEqual(len(set(SCREENING.CANDIDATE_REPRESENTATION_PAIRS)), 6)
        self.assertIn("hierarchical_v2", SCREENING.CANDIDATE_SINGLE_VIEWS)
        self.assertNotIn("hierarchical", SCREENING.CANDIDATE_SINGLE_VIEWS)

    def test_versioned_hierarchy_is_not_silently_part_of_frozen_oracle(self) -> None:
        self.assertNotIn("hierarchical_v2", SCREENING.SINGLE_VIEWS)
        self.assertIn("hierarchical_v2", SCREENING.VERSIONED_CANDIDATE_VIEWS)


if __name__ == "__main__":
    unittest.main()
