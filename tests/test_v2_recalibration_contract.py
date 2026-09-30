import importlib.util
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "v2_recalibration", ROOT / "tools" / "run_adaptive_single_view_v2_recalibration.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_fresh_sample_is_frozen_size_and_disjoint():
    sample, partitions, audit = MODULE.fresh_sample()
    assert len(sample) == 225
    assert set(partitions.values()) == {"tuning", "selection"}
    assert sum(value == "tuning" for value in partitions.values()) == 162
    assert sum(value == "selection" for value in partitions.values()) == 63
    assert audit["design_exposed_overlap"] == 0
    assert audit["heldout_id_overlap"] == 0
    assert audit["qasper_test_overlap_structural"] == 0
    assert audit["all_ids_are_training"] is True
    assert set(audit["stratum_counts"]) == MODULE.TARGET_STRATA


def test_grid_and_candidates_match_frozen_protocol():
    configs = MODULE.configurations()
    assert len(configs) == 1152
    assert {config.representation for config in configs} == set(MODULE.REPRESENTATIONS)
    assert {config.max_context_tokens for config in configs} == {800, 1200, 1600}
    assert all(config.probe_depth == 3 and config.adaptive_batch_size == 3 for config in configs)
    assert len({config.fingerprint for config in configs}) == len(configs)


def test_final_selection_uses_frozen_tie_order_only_after_metrics():
    rows = []
    for representation, recall, cer, tokens in (
        ("semantic", .70, .60, 900),
        ("paragraph", .709, .70, 1000),
        ("hierarchical_v2", .71, .65, 800),
    ):
        rows.append({"configuration_id": representation, "mode": "adaptive",
                     "representation": representation, "token_point": 1200,
                     "evidence_recall": recall, "answerable_CER": cer,
                     "context_tokens": tokens, "candidates_examined": 20})
    selected = [{"configuration_id": row["configuration_id"]} for row in rows]
    winner = MODULE.choose_final(rows, selected)
    assert winner["representation"] == "paragraph"


def test_recalibration_tool_has_no_groq_dependency():
    source = (ROOT / "tools" / "run_adaptive_single_view_v2_recalibration.py").read_text(encoding="utf-8")
    assert "from sar_rag.groq" not in source
    assert "GroqClient" not in source
