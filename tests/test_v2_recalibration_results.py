import csv
import json
import pathlib

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
RUN = ROOT / "output" / "adaptive_single_view_v2" / "recalibration_seed20260905"


def test_completed_run_counts_and_zero_llm():
    manifest = json.loads((RUN / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPLETE_RETRIEVAL_ONLY"
    assert manifest["llm_calls"] == 0
    assert manifest["failures"] == []
    assert manifest["overlap_audit"]["design_exposed_overlap"] == 0
    assert manifest["overlap_audit"]["heldout_id_overlap"] == 0
    with (RUN / "TUNING_PER_EXAMPLE.csv").open(encoding="utf-8", newline="") as stream:
        assert sum(1 for _ in csv.DictReader(stream)) == 162 * 1152
    with (RUN / "SELECTION_PER_EXAMPLE.csv").open(encoding="utf-8", newline="") as stream:
        assert sum(1 for _ in csv.DictReader(stream)) == 63 * 49


def test_frozen_winner_and_reproduction():
    manifest = json.loads((RUN / "FROZEN_V2_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["selected_configuration"]["representation"] == "paragraph"
    assert manifest["selected_configuration"]["max_context_tokens"] == 1600
    assert manifest["selection_metrics"]["evidence_recall"] == pytest.approx(0.7380952380952381)
    assert manifest["selection_metrics"]["answerable_CER"] == pytest.approx(0.5)
    assert manifest["selection_metrics"]["context_tokens"] == pytest.approx(1199.2063492063492)
    reproduction = json.loads((RUN / "REPRODUCTION_VALIDATION.json").read_text(encoding="utf-8"))
    assert reproduction["substantive_identical"] is True


def test_primary_comparisons_are_scoped():
    rows = list(csv.DictReader((RUN / "PAIRED_UNCERTAINTY.csv").open(encoding="utf-8", newline="")))
    lookup = {(row["metric"], row["reference"]): row for row in rows}
    assert float(lookup[("evidence_recall", "probe_only")]["difference"]) == pytest.approx(0.2923280423)
    assert float(lookup[("evidence_recall", "fixed_k10")]["difference"]) == pytest.approx(0.0714285714)
    assert float(lookup[("context_tokens", "fixed_k10")]["difference"]) > 0
    assert float(lookup[("evidence_recall", "fixed_k20")]["ci_low"]) >= -0.02
    assert float(lookup[("context_tokens", "fixed_k20")]["difference"]) < 0


def test_answerable_cer_and_residual_support_are_separate_columns():
    rows = list(csv.DictReader((RUN / "SELECTION_AGGREGATE.csv").open(encoding="utf-8", newline="")))
    winner = next(row for row in rows if row["configuration_id"] == "adaptive-c991560f562d12af41a1")
    assert float(winner["answerable_CER"]) == pytest.approx(0.5)
    assert float(winner["insufficient_residual_support_complete_recovery"]) == pytest.approx(17 / 21)


def test_overall_v2_replacement_claim_is_unsupported_by_pareto_audit():
    analysis = json.loads((RUN / "PARETO_AND_DECISION.json").read_text(encoding="utf-8"))
    assert analysis["overall_classification"] == "unsupported"
    assert analysis["selected_is_pareto_nondominated"] is False
    assert analysis["selected_is_nondominated_excluding_descriptive_v1"] is True
    selected = next(row for row in analysis["dominated_conditions"]
                    if row["configuration_id"] == analysis["selected_configuration_id"])
    assert selected["dominated_by"] == ["v1_full_frozen"]
