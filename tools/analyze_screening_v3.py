"""Validate and analyse the completed screening-v3 (n=286) retrieval-only experiment.

Recomputes every number directly from the per-example CSV. Never reads a summary
JSON as evidence. Makes no LLM call, changes no source module, touches no
evaluation split and never writes to the frozen calibration manifest.

Design decisions are taken ONLY from the untouched `selection` partition; the
`tuning` partition and the pooled sample are reported as descriptive context.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
from collections import Counter, defaultdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
V3 = ROOT / "output" / "independent_verification" / "screening_v3_n286"
OUT = ROOT / "output" / "independent_verification"

REFERENCE = ("sar_rag", "full")
CANDIDATE = ("sar_rag", "full_hierarchical_v2")

LEGACY_SINGLES = ["sentence_window_only", "paragraph_only", "semantic_only", "hierarchical_only"]
CAND_SINGLES = ["sentence_window_only", "paragraph_only", "semantic_only", "hierarchical_v2_only"]
LEGACY_PAIRS = ["pair_sentence_window__paragraph", "pair_sentence_window__semantic",
                "pair_sentence_window__hierarchical", "pair_paragraph__semantic",
                "pair_paragraph__hierarchical", "pair_semantic__hierarchical"]
CAND_PAIRS = ["candidate_pair_sentence_window__paragraph", "candidate_pair_sentence_window__semantic",
              "candidate_pair_sentence_window__hierarchical_v2", "candidate_pair_paragraph__semantic",
              "candidate_pair_paragraph__hierarchical_v2", "candidate_pair_semantic__hierarchical_v2"]
GATES = ["no_novelty", "no_similarity_gap", "no_post_retrieval_deduplication"]

OUTPUT_FIELDS = ["final_k", "context_tokens", "candidates_examined", "retrieval_rounds",
                 "stop_reason", "evidence_recall", "evidence_precision", "complete_evidence"]

# Frozen in SCREENING_V3_PROTOCOL.json before execution. Never edited after results.
MARGIN_RECALL_NONINFERIORITY = -0.02
MARGIN_CER_NONINFERIORITY = -0.02
MARGIN_MEANINGFUL_RECALL_GAIN = 0.02
MARGIN_CONTEXT_REDUCTION_PCT = 10.0
MARGIN_ORACLE2_GAIN = 0.05


def fnum(v: str) -> float:
    if v is None or v == "" or str(v).lower() == "nan":
        return float("nan")
    return float(v)


def paired(a: list[float], b: list[float], resamples: int, seed: int) -> dict:
    """Paired bootstrap CI and paired-randomization p for (b - a) over complete pairs."""
    pairs = [(x, y) for x, y in zip(a, b) if not math.isnan(x) and not math.isnan(y)]
    if len(pairs) < 2:
        return {"n": len(pairs)}
    d = np.asarray([y - x for x, y in pairs], float)
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), size=(resamples, len(d)))].mean(axis=1)
    signs = rng.integers(0, 2, size=(resamples, len(d))) * 2 - 1
    rand = (d * signs).mean(axis=1)
    obs = float(d.mean())
    sd = float(d.std(ddof=1))
    return {
        "n": len(pairs),
        "mean_reference": float(np.mean([x for x, _ in pairs])),
        "mean_condition": float(np.mean([y for _, y in pairs])),
        "difference": obs,
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "p_value": float((np.sum(np.abs(rand) >= abs(obs)) + 1) / (resamples + 1)),
        "dz": float(obs / sd) if sd > 0 else 0.0,
    }


def mcnemar(a: list[float], b: list[float]) -> dict:
    pairs = [(int(x), int(y)) for x, y in zip(a, b) if not math.isnan(x) and not math.isnan(y)]
    a_only = sum(x == 1 and y == 0 for x, y in pairs)
    b_only = sum(x == 0 and y == 1 for x, y in pairs)
    n = a_only + b_only
    if n == 0:
        return {"reference_only": 0, "condition_only": 0, "discordant": 0, "p_value": 1.0}
    s = min(a_only, b_only)
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(s + 1)) / 2 ** n)
    return {"reference_only": a_only, "condition_only": b_only, "discordant": n, "p_value": p}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values with monotonicity enforced."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted, running = {}, 0.0
    for rank, (key, p) in enumerate(items, start=1):
        running = max(running, min(1.0, (m - rank + 1) * p))
        adjusted[key] = running
    return adjusted


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resamples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260807)
    args = ap.parse_args()

    rows = list(csv.DictReader((V3 / "SCREENING_V2_PER_EXAMPLE.csv").open(encoding="utf-8")))

    # ---------------------------------------------------------------- validation
    by = defaultdict(dict)
    for r in rows:
        by[(r["arm"], r["condition"])][r["example_id"]] = r
    meta = {r["example_id"]: r for r in rows if (r["arm"], r["condition"]) == REFERENCE}
    ids_all = list(meta)
    ids = {
        "selection": [i for i in ids_all if meta[i]["design_partition"] == "selection"],
        "tuning": [i for i in ids_all if meta[i]["design_partition"] == "tuning"],
        "all": ids_all,
    }
    conditions = sorted(by)

    manifest_path = ROOT / "output" / "calibration" / "frozen_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cal_ids = set(manifest["sample_ids"])
    prior = set()
    for name in ("SCREENING_V2_PER_EXAMPLE.csv", "REPRESENTATION_ROUTING_PER_EXAMPLE.csv",
                 "screening_v3_smoke/SCREENING_V2_PER_EXAMPLE.csv"):
        p = OUT / name
        if p.exists():
            prior |= {r["example_id"] for r in csv.DictReader(p.open(encoding="utf-8"))}

    scored = [i for i in ids_all if float(meta[i]["gold_spans"]) > 0]
    validation = {
        "artifacts_present": sorted(p.name for p in V3.iterdir()),
        "row_count": len(rows),
        "unique_examples": len(ids_all),
        "condition_count": len(conditions),
        "rows_per_condition": sorted({len(v) for v in by.values()}),
        "expected_rows": len(ids_all) * len(conditions),
        "row_count_matches_expected": len(rows) == len(ids_all) * len(conditions),
        "duplicate_keys": sum(
            1 for v in Counter((r["example_id"], r["arm"], r["condition"]) for r in rows).values() if v > 1),
        "partition_sizes": dict(Counter(meta[i]["design_partition"] for i in ids_all)),
        "partition_matches_amendment": (
            sum(1 for i in ids_all if meta[i]["design_partition"] == "tuning") == 194
            and sum(1 for i in ids_all if meta[i]["design_partition"] == "selection") == 92),
        "dataset_sizes": dict(Counter(meta[i]["dataset"] for i in ids_all)),
        "stratum_sizes": dict(Counter(meta[i]["stratum"] for i in ids_all)),
        "frozen_calibration_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "frozen_calibration_manifest_expected_sha256":
            "5c570ec2b4c52e1b0ea0f2152a6f8fb6f1fa1c1575e58cfb58bde2c434a54d05",
        "calibration_id_overlap": len(cal_ids & set(ids_all)),
        "prior_diagnostic_id_overlap": len(prior & set(ids_all)),
        "split_tags_observed": dict(Counter(i.split(":")[1] for i in ids_all)),
        "evaluation_split_ids_present": sum(1 for i in ids_all if i.split(":")[1] != "train"),
        "examples_with_gold_evidence": len(scored),
        "examples_without_gold_evidence_undefined_recall": len(ids_all) - len(scored),
        "nan_rows_per_metric_expected": (len(ids_all) - len(scored)) * len(conditions),
        "nan_rows_per_metric_observed": {
            m: sum(1 for r in rows if math.isnan(fnum(r[m])))
            for m in ("evidence_recall", "complete_evidence", "evidence_precision")},
        "nan_rows_are_exactly_the_no_gold_examples": all(
            math.isnan(fnum(r[m])) == (float(r["gold_spans"]) == 0)
            for r in rows for m in ("evidence_recall", "complete_evidence")),
        "scored_examples_by_partition": {
            k: sum(1 for i in v if float(meta[i]["gold_spans"]) > 0) for k, v in ids.items()},
        "scored_examples_by_partition_dataset": {
            f"{k}|{d}": sum(1 for i in v if float(meta[i]["gold_spans"]) > 0 and meta[i]["dataset"] == d)
            for k, v in ids.items() for d in ("qasper", "musique_full")},
    }

    def series(key, metric, subset, dataset=None):
        d = by[key]
        return [fnum(d[i][metric]) if i in d else float("nan")
                for i in ids[subset]
                if dataset is None or meta[i]["dataset"] == dataset]

    # ---------------------------------------------------------------- comparisons
    results: list[dict] = []
    families: dict[tuple, dict[str, float]] = defaultdict(dict)

    for metric in ("evidence_recall", "complete_evidence", "evidence_precision", "context_tokens", "final_k"):
        for subset in ("selection", "tuning", "all"):
            for dataset in (None, "qasper", "musique_full"):
                ref = series(REFERENCE, metric, subset, dataset)
                for key in conditions:
                    if key == REFERENCE:
                        continue
                    st = paired(ref, series(key, metric, subset, dataset), args.resamples, args.seed)
                    if st.get("n", 0) < 2:
                        continue
                    row = {"metric": metric, "partition": subset,
                           "dataset": dataset or "both", "arm": key[0], "condition": key[1], **st}
                    if metric == "complete_evidence":
                        mc = mcnemar(ref, series(key, metric, subset, dataset))
                        row.update({f"mcnemar_{k}": v for k, v in mc.items()})
                    results.append(row)
                    families[(metric, subset, dataset or "both")][key[1]] = st["p_value"]

    adjusted = {fam: holm(ps) for fam, ps in families.items()}
    for row in results:
        fam = (row["metric"], row["partition"], row["dataset"])
        row["holm_p_all_conditions"] = adjusted[fam][row["condition"]]
        row["holm_family_size"] = len(families[fam])

    # transparent four-single-view family, computed separately
    for metric in ("evidence_recall", "complete_evidence"):
        for subset in ("selection", "tuning", "all"):
            for dataset in (None, "qasper", "musique_full"):
                for label, group in (("legacy", LEGACY_SINGLES), ("candidate", CAND_SINGLES)):
                    fam = {c: families[(metric, subset, dataset or "both")].get(c) for c in group}
                    fam = {k: v for k, v in fam.items() if v is not None}
                    if len(fam) < 2:
                        continue
                    adj = holm(fam)
                    for row in results:
                        if (row["metric"], row["partition"], row["dataset"]) == (metric, subset, dataset or "both") \
                                and row["condition"] in adj:
                            row[f"holm_p_four_single_view_{label}"] = adj[row["condition"]]

    # ---------------------------------------------------------------- oracles
    def oracle_series(members, metric, subset, dataset=None):
        out = []
        for i in ids[subset]:
            if dataset is not None and meta[i]["dataset"] != dataset:
                continue
            vals = [fnum(by[("sar_rag", c)][i][metric]) for c in members if i in by[("sar_rag", c)]]
            vals = [v for v in vals if not math.isnan(v)]
            out.append(max(vals) if vals else float("nan"))
        return out

    def maximizers(members, i, metric="evidence_recall"):
        vals = {c: fnum(by[("sar_rag", c)][i][metric]) for c in members if i in by[("sar_rag", c)]}
        vals = {k: v for k, v in vals.items() if not math.isnan(v)}
        if not vals:
            return []
        top = max(vals.values())
        return sorted(k for k, v in vals.items() if abs(v - top) < 1e-9)

    oracles = {}
    for subset in ("selection", "tuning", "all"):
        for dataset in (None, "qasper", "musique_full"):
            scope = f"{subset}|{dataset or 'both'}"
            block = {}
            for label, singles, pairs in (("legacy", LEGACY_SINGLES, LEGACY_PAIRS),
                                          ("candidate", CAND_SINGLES, CAND_PAIRS)):
                o1 = oracle_series(singles, "evidence_recall", subset, dataset)
                o2 = oracle_series(singles + pairs, "evidence_recall", subset, dataset)
                defined = [(a, b) for a, b in zip(o1, o2) if not math.isnan(a)]
                consts = {c: float(np.nanmean(series(("sar_rag", c), "evidence_recall", subset, dataset)))
                          for c in singles}
                best_const = max(consts, key=consts.get)
                router_key = REFERENCE if label == "legacy" else CANDIDATE
                router = float(np.nanmean(series(router_key, "evidence_recall", subset, dataset)))
                pair_means = {c: float(np.nanmean(series(("sar_rag", c), "evidence_recall", subset, dataset)))
                              for c in pairs}
                best_pair = max(pair_means, key=pair_means.get)
                tie_counts = Counter(
                    len(maximizers(singles + pairs, i))
                    for i in ids[subset]
                    if (dataset is None or meta[i]["dataset"] == dataset) and float(meta[i]["gold_spans"]) > 0)
                block[label] = {
                    "n_scored": len(defined),
                    "oracle1": float(np.mean([a for a, _ in defined])) if defined else float("nan"),
                    "oracle2": float(np.mean([b for _, b in defined])) if defined else float("nan"),
                    "oracle2_minus_oracle1": float(np.mean([b - a for a, b in defined])) if defined else float("nan"),
                    "constant_single_view_means": consts,
                    "best_constant_single_view": best_const,
                    "best_constant_single_view_recall": consts[best_const],
                    "constant_pair_means": pair_means,
                    "best_constant_pair": best_pair,
                    "best_constant_pair_recall": pair_means[best_pair],
                    "router_recall": router,
                    "oracle1_minus_best_constant": (
                        float(np.mean([a for a, _ in defined])) - consts[best_const]) if defined else float("nan"),
                    "oracle2_minus_best_constant": (
                        float(np.mean([b for _, b in defined])) - consts[best_const]) if defined else float("nan"),
                    "router_regret_vs_oracle1": (
                        float(np.mean([a for a, _ in defined])) - router) if defined else float("nan"),
                    "router_regret_vs_oracle2": (
                        float(np.mean([b for _, b in defined])) - router) if defined else float("nan"),
                    "router_minus_best_constant": router - consts[best_const],
                    "examples_with_tied_maximizers": sum(v for k, v in tie_counts.items() if k > 1),
                    "tie_multiplicity_histogram": dict(sorted(tie_counts.items())),
                }
            # paired inference for the oracle gaps that drive the decision rules
            for label, singles, pairs in (("legacy", LEGACY_SINGLES, LEGACY_PAIRS),
                                          ("candidate", CAND_SINGLES, CAND_PAIRS)):
                bc = block[label]["best_constant_single_view"]
                o2 = oracle_series(singles + pairs, "evidence_recall", subset, dataset)
                block[label]["oracle2_vs_best_constant_paired"] = paired(
                    series(("sar_rag", bc), "evidence_recall", subset, dataset), o2,
                    args.resamples, args.seed)
            oracles[scope] = block

    # ---------------------------------------------------------------- gate effects
    gate_report = {}
    for gate in GATES + ["select_one_representation"]:
        key = ("sar_rag", gate)
        changed = {f: 0 for f in OUTPUT_FIELDS}
        examples_changed = 0
        for i in ids_all:
            a, b = by[REFERENCE][i], by[key][i]
            any_change = False
            for f in OUTPUT_FIELDS:
                va, vb = a[f], b[f]
                if f in ("stop_reason",):
                    diff = va != vb
                else:
                    fa, fb = fnum(va), fnum(vb)
                    diff = not (math.isnan(fa) and math.isnan(fb)) and fa != fb
                if diff:
                    changed[f] += 1
                    any_change = True
            examples_changed += any_change
        sel = {m: next((r for r in results if r["condition"] == gate and r["metric"] == m
                        and r["partition"] == "selection" and r["dataset"] == "both"), None)
               for m in ("evidence_recall", "complete_evidence", "context_tokens")}
        gate_report[gate] = {
            "examples_with_any_output_change": examples_changed,
            "examples_total": len(ids_all),
            "changed_examples_by_field": changed,
            "selection_partition": {m: (None if v is None else
                                        {k: v[k] for k in ("n", "difference", "ci_low", "ci_high",
                                                           "p_value", "holm_p_all_conditions", "dz")})
                                    for m, v in sel.items()},
            "changes_any_final_output": examples_changed > 0,
            "recall_not_worse_than_margin": (
                sel["evidence_recall"]["difference"] >= MARGIN_RECALL_NONINFERIORITY
                if sel["evidence_recall"] else None),
            "retained_as_claimed_component": examples_changed > 0,
        }

    # ---------------------------------------------------------------- decision rules
    def sel_row(cond, metric, dataset="both", partition="selection"):
        return next((r for r in results if r["condition"] == cond and r["metric"] == metric
                     and r["partition"] == partition and r["dataset"] == dataset), None)

    o = oracles["selection|both"]
    cand_vs_full = sel_row("full_hierarchical_v2", "evidence_recall")
    cand_ctx = sel_row("full_hierarchical_v2", "context_tokens")
    h2_vs_h1 = paired(series(("sar_rag", "hierarchical_only"), "evidence_recall", "selection"),
                      series(("sar_rag", "hierarchical_v2_only"), "evidence_recall", "selection"),
                      args.resamples, args.seed)

    best_const = o["candidate"]["best_constant_single_view"]
    router_vs_bestconst = paired(
        series(("sar_rag", best_const), "evidence_recall", "selection"),
        series(CANDIDATE, "evidence_recall", "selection"), args.resamples, args.seed)
    legacy_best_const = o["legacy"]["best_constant_single_view"]
    legacy_router_vs_bestconst = paired(
        series(("sar_rag", legacy_best_const), "evidence_recall", "selection"),
        series(REFERENCE, "evidence_recall", "selection"), args.resamples, args.seed)
    ctx_vs_bestconst = paired(
        series(("sar_rag", best_const), "context_tokens", "selection"),
        series(CANDIDATE, "context_tokens", "selection"), args.resamples, args.seed)

    hier_rule = {
        "hierarchical_v2_improves_over_v1": h2_vs_h1["difference"] > 0,
        "hierarchical_v2_vs_v1": h2_vs_h1,
        "candidate_primary_noninferior_to_legacy": cand_vs_full["difference"] >= MARGIN_RECALL_NONINFERIORITY,
        "candidate_primary_vs_legacy_recall": cand_vs_full,
        "candidate_primary_vs_legacy_context": cand_ctx,
        "gain_recall_ge_margin": cand_vs_full["difference"] >= MARGIN_MEANINGFUL_RECALL_GAIN,
        "context_reduction_percent": (
            -100.0 * cand_ctx["difference"] / cand_ctx["mean_reference"] if cand_ctx["mean_reference"] else 0.0),
        "context_reduction_ge_margin": (
            -100.0 * cand_ctx["difference"] / cand_ctx["mean_reference"] >= MARGIN_CONTEXT_REDUCTION_PCT
            if cand_ctx["mean_reference"] else False),
        "oracle2_gain_over_legacy_representation_set": (
            o["candidate"]["oracle2"] - o["legacy"]["oracle2"]),
        "oracle2_gain_ge_margin": (
            o["candidate"]["oracle2"] - o["legacy"]["oracle2"] >= MARGIN_MEANINGFUL_RECALL_GAIN),
    }
    hier_rule["RETAINED"] = bool(
        hier_rule["hierarchical_v2_improves_over_v1"]
        and hier_rule["candidate_primary_noninferior_to_legacy"]
        and (hier_rule["gain_recall_ge_margin"] or hier_rule["context_reduction_ge_margin"]
             or hier_rule["oracle2_gain_ge_margin"]))

    routing_rule = {
        "best_constant_single_view": best_const,
        "candidate_router_vs_best_constant": router_vs_bestconst,
        "legacy_best_constant_single_view": legacy_best_const,
        "legacy_router_vs_best_constant": legacy_router_vs_bestconst,
        "noninferior_to_best_constant": router_vs_bestconst["difference"] >= MARGIN_RECALL_NONINFERIORITY,
        "gain_recall_ge_margin": router_vs_bestconst["difference"] >= MARGIN_MEANINGFUL_RECALL_GAIN,
        "context_vs_best_constant": ctx_vs_bestconst,
        "context_reduction_percent": (
            -100.0 * ctx_vs_bestconst["difference"] / ctx_vs_bestconst["mean_reference"]
            if ctx_vs_bestconst["mean_reference"] else 0.0),
        "context_reduction_ge_margin": (
            -100.0 * ctx_vs_bestconst["difference"] / ctx_vs_bestconst["mean_reference"]
            >= MARGIN_CONTEXT_REDUCTION_PCT if ctx_vs_bestconst["mean_reference"] else False),
        "oracle2_minus_best_constant": o["candidate"]["oracle2_minus_best_constant"],
        "oracle2_ceiling_ge_margin": o["candidate"]["oracle2_minus_best_constant"] >= MARGIN_ORACLE2_GAIN,
        "oracle2_vs_best_constant_paired": o["candidate"]["oracle2_vs_best_constant_paired"],
    }
    routing_rule["RETAINED"] = bool(
        routing_rule["noninferior_to_best_constant"]
        and (routing_rule["gain_recall_ge_margin"] or routing_rule["context_reduction_ge_margin"])
        and routing_rule["oracle2_ceiling_ge_margin"])

    trad = {}
    for cond in ["sliding_window_k3", "sliding_window_k5", "sliding_window_k10", "sliding_window_k20",
                 "sliding_window_matched_k3", "sliding_window_matched_k5",
                 "sliding_window_matched_k10", "sliding_window_matched_k20"]:
        rec = sel_row(cond, "evidence_recall")
        ctx = sel_row(cond, "context_tokens")
        trad[cond] = {
            "recall_difference_vs_full": rec["difference"], "recall_ci": [rec["ci_low"], rec["ci_high"]],
            "recall_p": rec["p_value"], "recall_holm_p": rec["holm_p_all_conditions"],
            "mean_context_tokens": ctx["mean_condition"],
            "reference_mean_context_tokens": ctx["mean_reference"],
            "sar_rag_beats_this": rec["difference"] < 0 and rec["holm_p_all_conditions"] < 0.05,
            "this_beats_sar_rag": rec["difference"] > 0 and rec["holm_p_all_conditions"] < 0.05,
        }
    budget_matched_defensible = not any(v["this_beats_sar_rag"] for v in trad.values())

    decision = {
        "hierarchy_candidate_retained": hier_rule,
        "routing_claim_retained": routing_rule,
        "gates": {k: v["retained_as_claimed_component"] for k, v in gate_report.items()},
        "traditional_rag": trad,
        "sar_rag_has_defensible_budget_matched_advantage": budget_matched_defensible,
    }
    if routing_rule["RETAINED"]:
        decision["FINAL"] = "RETAIN_CURRENT_ROUTING"
    elif (routing_rule["oracle2_ceiling_ge_margin"]
          and routing_rule["oracle2_vs_best_constant_paired"].get("p_value", 1.0) < 0.05):
        decision["FINAL"] = "REDESIGN_AND_RECALIBRATE_ROUTING"
    else:
        decision["FINAL"] = "PIVOT_TO_ADAPTIVE_SINGLE_VIEW"

    # ---------------------------------------------------------------- write
    (OUT / "SCREENING_V3_VALIDATION.json").write_text(json.dumps({
        "validation": validation,
        "oracles": oracles,
        "gate_instrumentation": gate_report,
        "decision_rules_applied": decision,
        "frozen_margins": {
            "evidence_recall_noninferiority": MARGIN_RECALL_NONINFERIORITY,
            "complete_evidence_noninferiority": MARGIN_CER_NONINFERIORITY,
            "meaningful_recall_gain": MARGIN_MEANINGFUL_RECALL_GAIN,
            "meaningful_context_reduction_percent": MARGIN_CONTEXT_REDUCTION_PCT,
            "meaningful_oracle2_gain_over_best_constant": MARGIN_ORACLE2_GAIN,
        },
    }, indent=2, default=float), encoding="utf-8")

    fields = ["metric", "partition", "dataset", "arm", "condition", "n", "mean_reference",
              "mean_condition", "difference", "ci_low", "ci_high", "p_value",
              "holm_p_all_conditions", "holm_family_size", "holm_p_four_single_view_legacy",
              "holm_p_four_single_view_candidate", "dz", "mcnemar_reference_only",
              "mcnemar_condition_only", "mcnemar_discordant", "mcnemar_p_value"]
    with (OUT / "SCREENING_V3_RESULTS.csv").open("w", newline="", encoding="utf-8") as s:
        w = csv.DictWriter(s, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(results, key=lambda r: (r["metric"], r["partition"], r["dataset"], r["condition"])))

    # ---------------------------------------------------------------- failures
    failures = []
    for i in ids_all:
        if float(meta[i]["gold_spans"]) == 0:
            continue
        r_full = fnum(by[REFERENCE][i]["evidence_recall"])
        best_single = max((fnum(by[("sar_rag", c)][i]["evidence_recall"]) for c in LEGACY_SINGLES))
        o2 = max((fnum(by[("sar_rag", c)][i]["evidence_recall"]) for c in CAND_SINGLES + CAND_PAIRS))
        failures.append({
            "example_id": i, "dataset": meta[i]["dataset"], "stratum": meta[i]["stratum"],
            "design_partition": meta[i]["design_partition"], "gold_spans": meta[i]["gold_spans"],
            "answerable": meta[i]["answerable"],
            "full_recall": r_full,
            "full_hierarchical_v2_recall": fnum(by[CANDIDATE][i]["evidence_recall"]),
            "best_single_view_recall": best_single,
            "oracle2_recall": o2,
            "routing_regret_vs_oracle2": o2 - r_full,
            "router_selected": by[REFERENCE][i]["selected_ordered"],
            "router_first_view": by[REFERENCE][i]["first_view"],
            "oracle2_maximizers": "|".join(maximizers(CAND_SINGLES + CAND_PAIRS, i)),
            "oracle2_tie_count": len(maximizers(CAND_SINGLES + CAND_PAIRS, i)),
            "full_stop_reason": by[REFERENCE][i]["stop_reason"],
            "full_final_k": by[REFERENCE][i]["final_k"],
            "full_context_tokens": by[REFERENCE][i]["context_tokens"],
            "sliding_window_k20_recall": fnum(by[("traditional", "sliding_window_k20")][i]["evidence_recall"]),
            "failure_class": ("total_miss" if r_full == 0 else
                              "partial" if r_full < 1.0 else "complete"),
            "recoverable_by_better_routing": int(o2 - r_full > 1e-9),
        })
    failures.sort(key=lambda r: (-r["routing_regret_vs_oracle2"], r["full_recall"]))
    with (OUT / "SCREENING_V3_FAILURES.csv").open("w", newline="", encoding="utf-8") as s:
        w = csv.DictWriter(s, fieldnames=list(failures[0]))
        w.writeheader()
        w.writerows(failures)

    print(json.dumps({
        "validation_ok": validation["row_count_matches_expected"] and validation["calibration_id_overlap"] == 0
                         and validation["evaluation_split_ids_present"] == 0,
        "results_rows": len(results), "failure_rows": len(failures),
        "decision": decision["FINAL"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
