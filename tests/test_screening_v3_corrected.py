import csv
import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "screening_v3_corrected", ROOT / "tools" / "analyze_screening_v3_corrected.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


@pytest.fixture(scope="module")
def analysis():
    source = ROOT / "output" / "independent_verification" / "screening_v3_n286" / "SCREENING_V2_PER_EXAMPLE.csv"
    rows = list(csv.DictReader(source.open(encoding="utf-8")))
    return MODULE.analyse(rows, resamples=2_000, seed=20260807)


def test_core_counts_and_frozen_manifest(analysis):
    _, validation = analysis
    assert validation["row_count"] == 9_724
    assert validation["unique_examples"] == 286
    assert validation["condition_count"] == 34
    assert validation["partition_counts"] == {"selection": 92, "tuning": 194}
    assert validation["evidence_scored_counts"] == {
        "all": 210, "tuning": 143, "selection": 67,
        "selection_answerable": 46,
        "selection_insufficient_with_residual_support": 21,
    }
    assert set(validation["nan_metric_cells"].values()) == {2_584}
    assert validation["frozen_v1_manifest_hash_matches_expected"] is True


def test_complete_evidence_populations_are_not_mixed(analysis):
    results, _ = analysis
    answerable = [row for row in results if row["metric"] == "answerable_complete_evidence_recovery"
                  and row["partition"] == "selection" and row["dataset"] == "both"]
    residual = [row for row in results
                if row["metric"] == "insufficient_residual_support_complete_recovery"
                and row["partition"] == "selection" and row["dataset"] == "both"]
    assert answerable and residual
    assert {row["n"] for row in answerable} == {46}
    assert {row["n"] for row in residual} == {21}


def test_candidate_and_legacy_oracle_complementarity_are_separate(analysis):
    _, validation = analysis
    candidate = validation["oracle_complementarity"]["hierarchical_v2_candidate"]
    legacy = validation["oracle_complementarity"]["legacy"]
    assert candidate["all|both"]["mean_pair_complementarity_gain"] == pytest.approx(0.0)
    assert candidate["all|both"]["examples_with_positive_pair_gain"] == 0
    assert legacy["selection|both"]["mean_pair_complementarity_gain"] == pytest.approx(0.0)
    assert legacy["all|both"]["examples_with_positive_pair_gain"] == 1
    assert legacy["all|both"]["mean_pair_complementarity_gain"] == pytest.approx(1 / 1470)
    assert legacy["all|both"]["maximum_example_pair_gain"] == pytest.approx(1 / 7)


def test_tokens_gates_stops_and_failure_reachability(analysis):
    _, validation = analysis
    assert validation["mean_context_tokens_selection"]["sar_rag:full"] == pytest.approx(1305.1521739130435)
    assert validation["stop_reasons_full_selection"] == {
        "all_selected_representations_paused": 58,
        "final_chunk_safety_cap": 28,
        "retrieval_round_safety_cap": 6,
    }
    # Novelty flags candidates, but its removal changes no final output because
    # the rejected candidates are also excluded by another active constraint.
    assert validation["gate_binding_counts"]["novelty_binding"] == 107
    assert validation["gate_binding_counts"]["utility_binding"] == 0
    assert validation["gate_binding_counts"]["redundancy_binding"] == 0
    assert validation["gate_binding_counts"]["gap_would_fire"] == 0
    assert validation["gate_output_change_counts"]["no_novelty"]["final_k"] == 0
    assert validation["gate_output_change_counts"]["no_similarity_gap"]["final_k"] == 0
    assert validation["gate_output_change_counts"]["no_post_retrieval_deduplication"]["final_k"] == 56
    assert validation["failure_reachability"]["original_full_all_scored"] == {
        "complete_misses": 26,
        "unreachable_by_any_candidate_single_or_pair": 18,
        "reachable_by_perfect_candidate_routing": 8,
    }
    assert validation["failure_reachability"]["hierarchical_v2_full_selection_scored"] == {
        "complete_misses": 10,
        "unreachable_by_any_candidate_single_or_pair": 9,
        "reachable_by_perfect_candidate_routing": 1,
    }


def test_budget_claim_requires_positive_matched_token_evidence(analysis):
    _, validation = analysis
    actual = validation["actual_token_comparison"]
    assert "traditional:sliding_window_k10" in actual["matched_traditional_operating_points"]
    assert actual["sar_has_significant_matched_actual_token_advantage"] is False


def test_pareto_front_uses_lower_tokens_and_higher_recall():
    front, dominated = MODULE.pareto_front({"a": (100, 0.5), "b": (100, 0.6), "c": (120, 0.55)})
    assert front == ["b"]
    assert {row["condition"] for row in dominated} == {"a", "c"}
