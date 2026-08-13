from __future__ import annotations

from dataclasses import replace

from .config import SarRagConfig


CONDITIONS = (
    "full",
    "sentence_window_only",
    "paragraph_only",
    "semantic_only",
    "hierarchical_only",
    "all_representations_no_rrf",
    "rrf_probe_only_no_adaptive",
    "no_novelty",
    "no_similarity_gap",
    "no_post_retrieval_deduplication",
    "select_one_representation",
    "probe_depth_5",
    "rrf_k0_1",
    "rrf_k0_10",
    "rrf_k0_60",
    "baseline_fixed_k_3",
    "baseline_fixed_k_5",
    "baseline_fixed_k_10",
    "baseline_fixed_k_20",
    "legacy_multi_call",
)


def apply_condition(config: SarRagConfig, static_k: int, condition: str) -> tuple[SarRagConfig, int, str]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown registered condition: {condition}")
    engine = "one_call"
    updates = {}
    if condition.endswith("_only") and condition != "select_one_representation":
        updates["enabled_representations"] = (condition.removesuffix("_only"),)
    elif condition == "all_representations_no_rrf":
        updates["representation_selection_mode"] = "all"
        updates["max_selected_representations"] = 2
    elif condition == "rrf_probe_only_no_adaptive":
        updates["adaptive_retrieval_enabled"] = False
    elif condition == "no_novelty":
        updates["novelty_enabled"] = False
        updates["novelty_threshold"] = 0.0
    elif condition == "no_similarity_gap":
        updates["similarity_gap_enabled"] = False
    elif condition == "no_post_retrieval_deduplication":
        updates["post_retrieval_deduplication_enabled"] = False
    elif condition == "select_one_representation":
        updates["max_selected_representations"] = 1
    elif condition == "probe_depth_5":
        updates["probe_depth"] = 5
    elif condition.startswith("rrf_k0_"):
        updates["rrf_k0"] = int(condition.rsplit("_", 1)[1])
    elif condition.startswith("baseline_fixed_k_"):
        static_k = int(condition.rsplit("_", 1)[1])
    elif condition == "legacy_multi_call":
        engine = "legacy_multi_call"
    return replace(config, **updates), static_k, engine


def registry() -> list[dict]:
    return [{"condition": name, "primary": name == "full",
             "online_policy": "multi_call_cost_ablation" if name == "legacy_multi_call" else "one_call"}
            for name in CONDITIONS]
