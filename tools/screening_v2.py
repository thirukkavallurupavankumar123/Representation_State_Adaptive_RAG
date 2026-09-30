"""Corrected retrieval-only screening: per-example, order-aware, instrumented.

Fixes five defects in tools/retrieval_only_screening.py found during review:
  1. Calibration IDs are now EXPLICITLY excluded (a different seed does not
     guarantee disjointness; the v1 screening sample shared 2 QASPER examples).
  2. Per-example rows are persisted so paired inference is possible at all.
  3. True selection ORDER is recovered from RepresentationProfile.selection_reasons
     (PipelineResult stores a sorted tuple, so v1's "first view" was alphabetical).
  4. Gate instrumentation separates evaluable / would-reject / actually-binding.
  5. Oracle-best-view and routing regret are computed per example.

Training data only. No Groq calls. Evaluation splits are never opened.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import replace
from itertools import chain
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sar_rag.ablations import apply_condition  # noqa: E402
from sar_rag.adaptive_pipeline import SarRagSystem  # noqa: E402
from sar_rag.benchmark import _documents  # noqa: E402
from sar_rag.calibration import (  # noqa: E402
    RetrievalOnlyAnswerer, _covered_spans, deterministic_stratified_sample,
    partition_training_sample, stratum,
)
from sar_rag.config import SarRagConfig  # noqa: E402
from sar_rag.corpus import build_corpus  # noqa: E402
from sar_rag.dataset_adapters import iter_musique_full, iter_qasper  # noqa: E402
from sar_rag.encoding import SentenceTransformerEncoder  # noqa: E402

SINGLE_VIEWS = ("sentence_window", "paragraph", "semantic", "hierarchical")
VERSIONED_CANDIDATE_VIEWS = ("hierarchical_v2",)
CANDIDATE_SINGLE_VIEWS = ("sentence_window", "paragraph", "semantic", "hierarchical_v2")
REPRESENTATION_PAIRS = tuple(
    (left, right)
    for position, left in enumerate(SINGLE_VIEWS)
    for right in SINGLE_VIEWS[position + 1:]
)
CANDIDATE_REPRESENTATION_PAIRS = tuple(
    (left, right)
    for position, left in enumerate(CANDIDATE_SINGLE_VIEWS)
    for right in CANDIDATE_SINGLE_VIEWS[position + 1:]
)
CONDITIONS = ("full", "select_one_representation", "no_novelty", "no_similarity_gap",
              "no_post_retrieval_deduplication", "all_representations_no_rrf",
              "rrf_probe_only_no_adaptive", "probe_depth_5",
              *(f"{v}_only" for v in SINGLE_VIEWS))
BASELINES = ("sliding_window", "sliding_window_matched")
BASELINE_KS = (3, 5, 10, 20)


def frozen() -> dict:
    m = json.loads((ROOT / "output" / "calibration" / "frozen_manifest.json").read_text(encoding="utf-8"))
    assert m["status"] == "training_only_calibration_frozen"
    return m


def ordered_selection(result) -> list[str]:
    """Recover TRUE selection order; PipelineResult only keeps a sorted tuple."""
    first, second = [], []
    for name, profile in result.representation_profiles.items():
        if not profile.selected:
            continue
        reasons = " ".join(profile.selection_reasons)
        if "highest_normalized_rrf_coverage" in reasons:
            first.append(name)
        elif "marginal_rrf_gain" in reasons:
            second.append(name)
        else:                      # ablation_all_representations
            second.append(name)
    return sorted(first) + sorted(second)


def gates(result, config: SarRagConfig) -> dict:
    """Independent per-gate statistics, not just the precedence-ordered reason."""
    scored = [e for e in result.trace if e.get("event") == "candidate_scored_without_llm"]
    batches = [e for e in result.trace if e.get("event") == "batch_evaluated_without_llm"]
    out = {
        "candidates_scored": len(scored),
        # Independent: would this gate reject, ignoring precedence?
        "relevance_would_reject": sum(e["normalized_relevance"] < config.similarity_threshold for e in scored),
        "novelty_would_reject": sum(e["novelty"] < config.novelty_threshold for e in scored),
        "utility_would_reject": sum(e["marginal_utility"] < config.marginal_utility_threshold for e in scored),
        # Reached: gate was actually evaluated (earlier gates passed).
        "novelty_reached": sum(e["normalized_relevance"] >= config.similarity_threshold for e in scored),
        # Binding: this gate is the recorded rejection reason.
        "relevance_binding": sum(e.get("rejection_reason") == "similarity_percentile" for e in scored),
        "novelty_binding": sum(e.get("rejection_reason") == "novelty" for e in scored),
        "utility_binding": sum(e.get("rejection_reason") == "marginal_utility" for e in scored),
        "redundancy_binding": sum(e.get("rejection_reason") == "redundancy" for e in scored),
        "budget_binding": sum(e.get("rejection_reason") == "context_or_chunk_budget" for e in scored),
        "accepted": sum(bool(e.get("accepted")) for e in scored),
        # Gap rule: eligible means the batch added nothing, so the rule was reachable.
        "batches": len(batches),
        "gap_eligible": sum(e["gap_rule_eligible"] for e in batches),
        "gap_would_fire": sum(e["gap_would_fire"] for e in batches),
        "gap_preempted_by_patience": sum(
            e["gap_would_fire"] and e["patience_would_fire"] for e in batches),
        "max_standardized_gap": max((e["standardized_gap"] for e in batches), default=float("nan")),
        "max_gap_on_empty_batch": max(
            (e["standardized_gap"] for e in batches if e["gap_rule_eligible"]), default=float("nan")),
        "max_canonical_overlap": max((e["canonical_overlap"] for e in scored), default=float("nan")),
        "max_overlap_times_similarity": max(
            (e["canonical_overlap"] * max(e["semantic_similarity_to_selected"], 0.0) for e in scored),
            default=float("nan")),
    }
    return out


def metrics(result, row: dict) -> dict:
    gold = tuple(row["gold_evidence_spans"])
    done = next(e for e in reversed(result.trace) if e.get("event") == "run_completed")
    order = ordered_selection(result)
    out = {
        "final_k": float(result.final_k),
        "context_tokens": float(done.get("final_context_tokens", 0)),
        "candidates_examined": float(done.get("candidates_examined", 0)),
        "retrieval_rounds": float(done.get("retrieval_rounds", 0)),
        "stop_reason": result.stop_reason,
        "selected_ordered": ">".join(order),
        "first_view": order[0] if order else "",
        "second_view": order[1] if len(order) > 1 else "",
        "selected_count": float(len(order)),
    }
    if not gold:
        out.update(evidence_recall=float("nan"), evidence_precision=float("nan"),
                   complete_evidence=float("nan"))
        return out
    covered, relevant = _covered_spans(result.final_evidence, gold)
    out.update(
        evidence_recall=len(covered) / len(gold),
        evidence_precision=sum(relevant) / len(relevant) if relevant else 0.0,
        complete_evidence=float(len(covered) == len(gold)),
    )
    return out


def baseline_metrics(artifacts, encoder, row, rep, k, budget) -> dict:
    cands = artifacts.indexes[rep].search(row["query"], 1, k, "adaptive")
    ev, used = [], 0
    for c in cands:
        chunk = artifacts.chunks_by_id[c.chunk_id]
        t = encoder.token_count(chunk.evidence_text)
        if used + t <= budget:
            ev.append(type("I", (), {"chunk": chunk})())
            used += t
    gold = tuple(row["gold_evidence_spans"])
    out = {"final_k": float(len(ev)), "context_tokens": float(used),
           "candidates_examined": float(len(cands)), "retrieval_rounds": 1.0,
           "stop_reason": "static_top_k", "selected_ordered": rep,
           "first_view": rep, "second_view": "", "selected_count": 1.0}
    if not gold:
        out.update(evidence_recall=float("nan"), evidence_precision=float("nan"),
                   complete_evidence=float("nan"))
        return out
    covered, relevant = _covered_spans(ev, gold)
    out.update(evidence_recall=len(covered) / len(gold),
               evidence_precision=sum(relevant) / len(relevant) if relevant else 0.0,
               complete_evidence=float(len(covered) == len(gold)))
    return out


def paired(a: list[float], b: list[float], resamples: int, seed: int) -> dict:
    """Paired bootstrap CI + randomization p for (b - a) over complete pairs."""
    pairs = [(x, y) for x, y in zip(a, b)
             if not math.isnan(float(x)) and not math.isnan(float(y))]
    if len(pairs) < 2:
        return {"n": len(pairs)}
    d = np.asarray([y - x for x, y in pairs], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(resamples, len(d)))
    boot = d[idx].mean(axis=1)
    signs = rng.integers(0, 2, size=(resamples, len(d))) * 2 - 1
    rand = (d * signs).mean(axis=1)
    obs = float(d.mean())
    return {
        "n": len(pairs), "mean_a": float(np.mean([x for x, _ in pairs])),
        "mean_b": float(np.mean([y for _, y in pairs])), "difference": obs,
        "ci_low": float(np.quantile(boot, .025)), "ci_high": float(np.quantile(boot, .975)),
        "p_value": float((np.sum(np.abs(rand) >= abs(obs)) + 1) / (resamples + 1)),
        "dz": float(obs / d.std(ddof=1)) if d.std(ddof=1) > 0 else 0.0,
    }


def mcnemar(a: list[float], b: list[float]) -> dict:
    pairs = [(int(x), int(y)) for x, y in zip(a, b)
             if not math.isnan(float(x)) and not math.isnan(float(y))]
    a_only = sum(x == 1 and y == 0 for x, y in pairs)
    b_only = sum(x == 0 and y == 1 for x, y in pairs)
    n = a_only + b_only
    if n == 0:
        return {"a_only": 0, "b_only": 0, "p_value": 1.0}
    s = min(a_only, b_only)
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(s + 1)) / 2 ** n)
    return {"a_only": a_only, "b_only": b_only, "discordant": n, "p_value": p}


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values for one declared comparison family."""
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    family_size = len(ordered)
    for position, (name, p_value) in enumerate(ordered):
        running = max(running, min(1.0, float(p_value) * (family_size - position)))
        adjusted[name] = running
    return adjusted


def add_holm_fields(comparisons: dict[str, dict]) -> None:
    """Attach correction for all arms and the predeclared single-view family.

    Screening is exploratory, but reporting raw p-values alone made the v2
    semantic-only result look confirmatory.  Families are separated by metric
    and dataset scope; the four single-view ablations also receive their own
    transparent sensitivity family.
    """
    families: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for name, result in comparisons.items():
        metric, scope, comparison = name.split("|", 2)
        families[(metric, scope)][comparison] = float(result["p_value"])
    for (metric, scope), values in families.items():
        adjusted_all = holm_adjust(values)
        single_names = {
            f"{view}_only_vs_full": values[f"{view}_only_vs_full"]
            for view in SINGLE_VIEWS
            if f"{view}_only_vs_full" in values
        }
        adjusted_single = holm_adjust(single_names)
        candidate_single_names = {
            f"{view}_only_vs_full": values[f"{view}_only_vs_full"]
            for view in CANDIDATE_SINGLE_VIEWS
            if f"{view}_only_vs_full" in values
        }
        adjusted_candidate_single = holm_adjust(candidate_single_names)
        for comparison, adjusted in adjusted_all.items():
            result = comparisons[f"{metric}|{scope}|{comparison}"]
            result["holm_p_all_conditions"] = adjusted
            result["comparison_family_all_n"] = len(values)
            if comparison in adjusted_single:
                result["holm_p_single_view_family"] = adjusted_single[comparison]
                result["comparison_family_single_views_n"] = len(single_names)
            if comparison in adjusted_candidate_single:
                result["holm_p_candidate_single_view_family"] = adjusted_candidate_single[comparison]
                result["comparison_family_candidate_single_views_n"] = len(candidate_single_names)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-stratum", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260805)
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "output" / "independent_verification")
    ap.add_argument(
        "--exclude-csv", type=Path, action="append", default=[],
        help="CSV containing example_id values already used for design/diagnosis; repeatable",
    )
    args = ap.parse_args()

    manifest = frozen()
    excluded = set(manifest["sample_ids"])
    exclusion_sources = ["output/calibration/frozen_manifest.json"]
    for csv_path in args.exclude_csv:
        with csv_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or "example_id" not in reader.fieldnames:
                raise ValueError(f"exclusion CSV lacks example_id: {csv_path}")
            excluded.update(row["example_id"] for row in reader if row.get("example_id"))
        exclusion_sources.append(str(csv_path))
    base = SarRagConfig(**manifest["frozen_configuration"]["proposed"])

    raw = ROOT / "datasets" / "raw"
    rows = chain(iter_qasper(raw / "qasper/data/qasper-train-v0.3.json", "train"),
                 iter_musique_full(raw / "musique/data/musique_full_v1.0_train.jsonl", "train"))
    # FIX 1: exclude calibration IDs BEFORE sampling.
    sample, population = deterministic_stratified_sample(
        (r for r in rows if r["example_id"] not in excluded), args.per_stratum, args.seed)
    leaked = excluded & {r["example_id"] for r in sample}
    assert not leaked, f"calibration leakage: {leaked}"
    partitions = partition_training_sample(sample, args.seed, 0.32)
    print(json.dumps({"stage": "sampled", "n": len(sample), "excluded_ids": len(excluded),
                      "exclusion_sources": exclusion_sources, "leakage": 0}), flush=True)

    encoder = SentenceTransformerEncoder(str(ROOT / "models" / "all-MiniLM-L6-v2"))
    answerer = RetrievalOnlyAnswerer()
    configs = {}
    for name in CONDITIONS:
        cfg, _, engine = apply_condition(base, 20, name)
        assert engine == "one_call"
        configs[name] = cfg
    # Oracle@2 candidates force each pair to be consumed under the same shared
    # context/chunk/candidate budgets.  Gold labels are used only later to take
    # the per-question maximum; they never enter retrieval.
    for left, right in REPRESENTATION_PAIRS:
        configs[f"pair_{left}__{right}"] = replace(
            base,
            enabled_representations=(left, right),
            representation_selection_mode="all",
            max_selected_representations=2,
        )
    for view in VERSIONED_CANDIDATE_VIEWS:
        configs[f"{view}_only"] = replace(base, enabled_representations=(view,))
    configs["full_hierarchical_v2"] = replace(
        base, enabled_representations=CANDIDATE_SINGLE_VIEWS,
    )
    for left, right in CANDIDATE_REPRESENTATION_PAIRS:
        configs[f"candidate_pair_{left}__{right}"] = replace(
            base,
            enabled_representations=(left, right),
            representation_selection_mode="all",
            max_selected_representations=2,
        )
    all_views = tuple(sorted({v for c in configs.values() for v in c.enabled_representations}))

    def restrict(art, views):
        if set(views) == set(art.indexes):
            return art
        idx = {n: art.indexes[n] for n in views}
        ch = {c.chunk_id: c for i in idx.values() for c in i.chunks}
        return type(art)(art.units_by_id, ch, idx)

    per_example: list[dict] = []
    gate_rows: list[dict] = []
    for pos, row in enumerate(sample, 1):
        docs = _documents(row)
        art = build_corpus(docs, encoder, all_views)
        bart = build_corpus(docs, encoder, BASELINES)
        common = {"example_id": row["example_id"], "dataset": row["dataset"],
                  "stratum": stratum(row), "design_partition": partitions[row["example_id"]],
                  "gold_spans": len(row["gold_evidence_spans"]),
                  "answerable": int(row.get("gold_answerability") == "ANSWERABLE")}
        for name, cfg in configs.items():
            res = SarRagSystem(restrict(art, cfg.enabled_representations),
                               encoder, answerer, cfg).ask(row["query"])
            per_example.append({**common, "arm": "sar_rag", "condition": name,
                                **metrics(res, row)})
            if name == "full":
                gate_rows.append({**common, **gates(res, cfg)})
        for rep in BASELINES:
            for k in BASELINE_KS:
                per_example.append({**common, "arm": "traditional", "condition": f"{rep}_k{k}",
                                    **baseline_metrics(bart, encoder, row, rep, k,
                                                       base.max_final_context_tokens)})
        print(json.dumps({"stage": "screened", "completed": pos, "total": len(sample)}), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(per_example[0])
    with (args.output_dir / "SCREENING_V2_PER_EXAMPLE.csv").open("w", newline="", encoding="utf-8") as s:
        w = csv.DictWriter(s, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(per_example)
    with (args.output_dir / "GATE_INSTRUMENTATION_PER_EXAMPLE.csv").open("w", newline="", encoding="utf-8") as s:
        w = csv.DictWriter(s, fieldnames=list(gate_rows[0]), extrasaction="ignore")
        w.writeheader(); w.writerows(gate_rows)

    # ---- paired inference vs the primary condition, overall and per dataset ----
    by = defaultdict(dict)
    for r in per_example:
        by[(r["arm"], r["condition"])][r["example_id"]] = r
    ids = [r["example_id"] for r in sample]
    ds_of = {r["example_id"]: r["dataset"] for r in sample}
    partition_of = partitions

    def series(key, metric, subset=None):
        d = by[key]
        return [float(d[i][metric]) if i in d else float("nan")
                for i in ids if (
                    subset is None or ds_of[i] == subset or partition_of[i] == subset
                )]

    comparisons = {}
    for metric in ("evidence_recall", "complete_evidence", "evidence_precision", "context_tokens"):
        for scope in (None, "qasper", "musique_full", "tuning", "selection"):
            ref = series(("sar_rag", "full"), metric, scope)
            for key in list(by):
                if key == ("sar_rag", "full"):
                    continue
                res = paired(ref, series(key, metric, scope), args.resamples, args.seed)
                if res.get("n", 0) >= 2:
                    comparisons[f"{metric}|{scope or 'all'}|{key[1]}_vs_full"] = res
    add_holm_fields(comparisons)
    cer_mc = {}
    for key in list(by):
        if key == ("sar_rag", "full"):
            continue
        cer_mc[f"{key[1]}_vs_full"] = mcnemar(
            series(("sar_rag", "full"), "complete_evidence"), series(key, "complete_evidence"))
    cer_holm_all = holm_adjust({name: result["p_value"] for name, result in cer_mc.items()})
    cer_single_names = {
        f"{view}_only_vs_full": cer_mc[f"{view}_only_vs_full"]["p_value"]
        for view in SINGLE_VIEWS
    }
    cer_holm_single = holm_adjust(cer_single_names)
    cer_candidate_single_names = {
        f"{view}_only_vs_full": cer_mc[f"{view}_only_vs_full"]["p_value"]
        for view in CANDIDATE_SINGLE_VIEWS
    }
    cer_holm_candidate_single = holm_adjust(cer_candidate_single_names)
    for name, result in cer_mc.items():
        result["holm_p_all_conditions"] = cer_holm_all[name]
        result["comparison_family_all_n"] = len(cer_mc)
        if name in cer_holm_single:
            result["holm_p_single_view_family"] = cer_holm_single[name]
            result["comparison_family_single_views_n"] = len(cer_single_names)
        if name in cer_holm_candidate_single:
            result["holm_p_candidate_single_view_family"] = cer_holm_candidate_single[name]
            result["comparison_family_candidate_single_views_n"] = len(cer_candidate_single_names)

    # ---- label-only diagnostic oracles; never available to the live router ----
    oracle = []
    for i in ids:
        single_values = {
            v: by[("sar_rag", f"{v}_only")].get(i, {}).get("evidence_recall", float("nan"))
            for v in SINGLE_VIEWS
        }
        singles = {
            key: float(value) for key, value in single_values.items()
            if not math.isnan(float(value))
        }
        pair_values = {
            f"{left}+{right}": by[("sar_rag", f"pair_{left}__{right}")]
            .get(i, {}).get("evidence_recall", float("nan"))
            for left, right in REPRESENTATION_PAIRS
        }
        pairs = {
            key: float(value) for key, value in pair_values.items()
            if not math.isnan(float(value))
        }
        candidate_single_values = {
            view: by[("sar_rag", f"{view}_only")].get(i, {}).get(
                "evidence_recall", float("nan")
            )
            for view in CANDIDATE_SINGLE_VIEWS
        }
        candidate_singles = {
            key: float(value) for key, value in candidate_single_values.items()
            if not math.isnan(float(value))
        }
        candidate_pair_values = {
            f"{left}+{right}": by[("sar_rag", f"candidate_pair_{left}__{right}")]
            .get(i, {}).get("evidence_recall", float("nan"))
            for left, right in CANDIDATE_REPRESENTATION_PAIRS
        }
        candidate_pairs = {
            key: float(value) for key, value in candidate_pair_values.items()
            if not math.isnan(float(value))
        }
        if not singles:
            continue
        oracle1 = max(singles.values())
        optimal_single_views = sorted(
            name for name, value in singles.items() if abs(value - oracle1) <= 1e-12
        )
        all_states = {**singles, **pairs}
        oracle2 = max(all_states.values())
        optimal_states = sorted(
            name for name, value in all_states.items() if abs(value - oracle2) <= 1e-12
        )
        candidate_oracle1 = max(candidate_singles.values())
        candidate_optimal_single_views = sorted(
            name for name, value in candidate_singles.items()
            if abs(value - candidate_oracle1) <= 1e-12
        )
        candidate_states = {**candidate_singles, **candidate_pairs}
        candidate_oracle2 = max(candidate_states.values())
        candidate_optimal_states = sorted(
            name for name, value in candidate_states.items()
            if abs(value - candidate_oracle2) <= 1e-12
        )
        router = float(by[("sar_rag", "full")][i]["evidence_recall"])
        candidate_router = float(
            by[("sar_rag", "full_hierarchical_v2")][i]["evidence_recall"]
        )
        para = singles.get("paragraph", float("nan"))
        semantic = singles.get("semantic", float("nan"))
        selected = tuple(
            value for value in by[("sar_rag", "full")][i]["selected_ordered"].split(">")
            if value
        )
        oracle.append({
            "example_id": i,
            "dataset": ds_of[i],
            "optimal_single_views": "+".join(optimal_single_views),
            "optimal_single_tie_count": len(optimal_single_views),
            "oracle1_recall": oracle1,
            "optimal_one_or_two_states": "|".join(optimal_states),
            "optimal_state_tie_count": len(optimal_states),
            "oracle2_recall": oracle2,
            "router_recall": router,
            "paragraph_recall": para,
            "semantic_recall": semantic,
            "regret_vs_oracle1": oracle1 - router,
            "regret_vs_oracle2": oracle2 - router,
            "router_selected_ordered": ">".join(selected),
            "router_first_is_oracle1_optimal": int(bool(selected) and selected[0] in optimal_single_views),
            "router_matches_oracle1_value": int(abs(router - oracle1) <= 1e-12),
            "router_matches_oracle2_value": int(abs(router - oracle2) <= 1e-12),
            "oracle1_beats_paragraph": int(oracle1 > para + 1e-12),
            "oracle2_beats_best_single": int(oracle2 > oracle1 + 1e-12),
            "candidate_optimal_single_views": "+".join(candidate_optimal_single_views),
            "candidate_optimal_single_tie_count": len(candidate_optimal_single_views),
            "candidate_oracle1_recall": candidate_oracle1,
            "candidate_optimal_one_or_two_states": "|".join(candidate_optimal_states),
            "candidate_optimal_state_tie_count": len(candidate_optimal_states),
            "candidate_oracle2_recall": candidate_oracle2,
            "candidate_router_recall": candidate_router,
            "candidate_regret_vs_oracle1": candidate_oracle1 - candidate_router,
            "candidate_regret_vs_oracle2": candidate_oracle2 - candidate_router,
            "candidate_oracle2_beats_best_single": int(
                candidate_oracle2 > candidate_oracle1 + 1e-12
            ),
            "candidate_oracle2_gain_over_legacy": candidate_oracle2 - oracle2,
        })
    with (args.output_dir / "ORACLE_ROUTING_REGRET.csv").open("w", newline="", encoding="utf-8") as s:
        w = csv.DictWriter(s, fieldnames=list(oracle[0])); w.writeheader(); w.writerows(oracle)

    def agg(items):
        if not items:
            return {"n": 0}
        return {
            "n": len(items),
            "oracle1_recall": float(np.mean([x["oracle1_recall"] for x in items])),
            "oracle2_recall": float(np.mean([x["oracle2_recall"] for x in items])),
            "router_recall": float(np.mean([x["router_recall"] for x in items])),
            "paragraph_recall": float(np.mean([x["paragraph_recall"] for x in items])),
            "semantic_recall": float(np.mean([x["semantic_recall"] for x in items])),
            "mean_regret_vs_oracle1": float(np.mean([x["regret_vs_oracle1"] for x in items])),
            "mean_regret_vs_oracle2": float(np.mean([x["regret_vs_oracle2"] for x in items])),
            "router_first_is_oracle1_optimal_percent": 100 * float(np.mean(
                [x["router_first_is_oracle1_optimal"] for x in items]
            )),
            "router_matches_oracle1_value_percent": 100 * float(np.mean(
                [x["router_matches_oracle1_value"] for x in items]
            )),
            "router_matches_oracle2_value_percent": 100 * float(np.mean(
                [x["router_matches_oracle2_value"] for x in items]
            )),
            "oracle2_beats_best_single_percent": 100 * float(np.mean(
                [x["oracle2_beats_best_single"] for x in items]
            )),
            "mean_oracle1_tie_count": float(np.mean(
                [x["optimal_single_tie_count"] for x in items]
            )),
            "candidate_oracle1_recall": float(np.mean(
                [x["candidate_oracle1_recall"] for x in items]
            )),
            "candidate_oracle2_recall": float(np.mean(
                [x["candidate_oracle2_recall"] for x in items]
            )),
            "candidate_router_recall": float(np.mean(
                [x["candidate_router_recall"] for x in items]
            )),
            "candidate_mean_regret_vs_oracle1": float(np.mean(
                [x["candidate_regret_vs_oracle1"] for x in items]
            )),
            "candidate_mean_regret_vs_oracle2": float(np.mean(
                [x["candidate_regret_vs_oracle2"] for x in items]
            )),
            "candidate_oracle2_beats_best_single_percent": 100 * float(np.mean(
                [x["candidate_oracle2_beats_best_single"] for x in items]
            )),
            "candidate_oracle2_gain_over_legacy": float(np.mean(
                [x["candidate_oracle2_gain_over_legacy"] for x in items]
            )),
        }

    gate_totals = {}
    for key in gate_rows[0]:
        if key in ("example_id", "dataset", "stratum", "design_partition"):
            continue
        values = [float(row[key]) for row in gate_rows if not math.isnan(float(row[key]))]
        gate_totals[key] = (
            max(values) if key.startswith("max_") else float(sum(values))
        ) if values else float("nan")
    summary = {
        "scope": "training-only, retrieval-only, zero Groq calls; evaluation splits untouched",
        "sample": {"n": len(sample), "seed": args.seed, "per_stratum": args.per_stratum,
                   "excluded_ids": len(excluded), "exclusion_sources": exclusion_sources,
                   "verified_leakage": 0,
                   "partitions": dict(Counter(partitions.values())),
                   "selection_fraction_target": 0.32},
        "oracle_routing": {"all": agg(oracle),
                           "qasper": agg([x for x in oracle if x["dataset"] == "qasper"]),
                           "musique_full": agg([x for x in oracle if x["dataset"] == "musique_full"]),
                           "tuning": agg([
                               x for x in oracle if partitions[x["example_id"]] == "tuning"
                           ]),
                           "selection": agg([
                               x for x in oracle if partitions[x["example_id"]] == "selection"
                           ])},
        "gate_instrumentation_totals": gate_totals,
        "selection_order_true": dict(Counter(
            r["selected_ordered"] for r in per_example
            if r["arm"] == "sar_rag" and r["condition"] == "full").most_common()),
        "first_view_true": dict(Counter(
            r["first_view"] for r in per_example
            if r["arm"] == "sar_rag" and r["condition"] == "full").most_common()),
        "paired_comparisons_vs_full": comparisons,
        "mcnemar_complete_evidence_vs_full": cer_mc,
    }
    (args.output_dir / "SCREENING_V2_INFERENCE.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "rows": len(per_example)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
