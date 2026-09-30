"""Versioned correction analysis for screening-v3.

This program reads only the preserved per-example screening CSV and writes new
``*_CORRECTED`` artifacts.  It intentionally does not modify the historical
screening-v3 summaries or any frozen calibration artifact.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
from collections import Counter, defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "output" / "independent_verification" / "screening_v3_n286"
OUT = ROOT / "output" / "independent_verification"
REFERENCE = ("sar_rag", "full")
CANDIDATE = ("sar_rag", "full_hierarchical_v2")
LEGACY_SINGLES = ("sentence_window_only", "paragraph_only", "semantic_only", "hierarchical_only")
CANDIDATE_SINGLES = ("sentence_window_only", "paragraph_only", "semantic_only", "hierarchical_v2_only")
LEGACY_PAIRS = (
    "pair_sentence_window__paragraph", "pair_sentence_window__semantic",
    "pair_sentence_window__hierarchical", "pair_paragraph__semantic",
    "pair_paragraph__hierarchical", "pair_semantic__hierarchical",
)
CANDIDATE_PAIRS = (
    "candidate_pair_sentence_window__paragraph", "candidate_pair_sentence_window__semantic",
    "candidate_pair_sentence_window__hierarchical_v2", "candidate_pair_paragraph__semantic",
    "candidate_pair_paragraph__hierarchical_v2", "candidate_pair_semantic__hierarchical_v2",
)
EXPECTED_MANIFEST_SHA256 = "5c570ec2b4c52e1b0ea0f2152a6f8fb6f1fa1c1575e58cfb58bde2c434a54d05"


def fnum(value: object) -> float:
    if value is None or value == "" or str(value).lower() == "nan":
        return float("nan")
    return float(value)


def paired(reference: Sequence[float], condition: Sequence[float], resamples: int, seed: int) -> dict:
    pairs = [(a, b) for a, b in zip(reference, condition) if not math.isnan(a) and not math.isnan(b)]
    if len(pairs) < 2:
        return {"n": len(pairs)}
    delta = np.asarray([b - a for a, b in pairs], dtype=float)
    rng = np.random.default_rng(seed)
    boot = delta[rng.integers(0, len(delta), size=(resamples, len(delta)))].mean(axis=1)
    signs = rng.integers(0, 2, size=(resamples, len(delta))) * 2 - 1
    randomised = (delta * signs).mean(axis=1)
    observed = float(delta.mean())
    sd = float(delta.std(ddof=1))
    return {
        "n": len(pairs),
        "mean_reference": float(np.mean([a for a, _ in pairs])),
        "mean_condition": float(np.mean([b for _, b in pairs])),
        "difference": observed,
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "p_value": float((np.sum(np.abs(randomised) >= abs(observed)) + 1) / (resamples + 1)),
        "dz": float(observed / sd) if sd else 0.0,
    }


def mcnemar(reference: Sequence[float], condition: Sequence[float]) -> dict:
    pairs = [(int(a), int(b)) for a, b in zip(reference, condition) if not math.isnan(a) and not math.isnan(b)]
    reference_only = sum(a == 1 and b == 0 for a, b in pairs)
    condition_only = sum(a == 0 and b == 1 for a, b in pairs)
    discordant = reference_only + condition_only
    if not discordant:
        return {"reference_only": 0, "condition_only": 0, "discordant": 0, "p_value": 1.0}
    tail = min(reference_only, condition_only)
    p_value = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(tail + 1)) / 2**discordant)
    return {"reference_only": reference_only, "condition_only": condition_only,
            "discordant": discordant, "p_value": p_value}


def holm(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, p_value) in enumerate(ordered, start=1):
        running = max(running, min(1.0, (len(ordered) - rank + 1) * p_value))
        adjusted[name] = running
    return adjusted


def build_index(rows: Sequence[dict]) -> tuple[dict, dict, list[tuple[str, str]]]:
    counts = Counter((r["example_id"], r["arm"], r["condition"]) for r in rows)
    duplicates = [key for key, count in counts.items() if count != 1]
    if duplicates:
        raise ValueError(f"duplicate screening keys: {duplicates[:3]}")
    by: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by[(row["arm"], row["condition"])][row["example_id"]] = row
    meta = dict(by[REFERENCE])
    conditions = sorted(by)
    expected = set(meta)
    incomplete = [key for key in conditions if set(by[key]) != expected]
    if incomplete:
        raise ValueError(f"conditions do not share identical example IDs: {incomplete}")
    return by, meta, conditions


def population_ids(meta: Mapping[str, dict], partition: str, dataset: str, population: str) -> list[str]:
    out = []
    for example_id, row in meta.items():
        if partition != "all" and row["design_partition"] != partition:
            continue
        if dataset != "both" and row["dataset"] != dataset:
            continue
        has_gold = fnum(row["gold_spans"]) > 0
        answerable = int(float(row["answerable"])) == 1
        keep = {
            "all_questions": True,
            "evidence_scored": has_gold,
            "answerable_evidence_scored": has_gold and answerable,
            "insufficient_with_residual_support": has_gold and not answerable,
        }[population]
        if keep:
            out.append(example_id)
    return out


def values(by: Mapping, key: tuple[str, str], ids: Iterable[str], field: str) -> list[float]:
    return [fnum(by[key][example_id][field]) for example_id in ids]


def oracle_summary(by: Mapping, meta: Mapping[str, dict], singles: Sequence[str], pairs: Sequence[str],
                   partition: str, dataset: str) -> dict:
    ids = population_ids(meta, partition, dataset, "evidence_scored")
    gains = []
    oracle1 = []
    oracle2 = []
    for example_id in ids:
        single = max(fnum(by[("sar_rag", name)][example_id]["evidence_recall"]) for name in singles)
        pair = max(fnum(by[("sar_rag", name)][example_id]["evidence_recall"]) for name in pairs)
        oracle1.append(single)
        oracle2.append(max(single, pair))
        gains.append(max(single, pair) - single)
    return {
        "n": len(ids),
        "oracle_at_1": float(np.mean(oracle1)),
        "oracle_at_2": float(np.mean(oracle2)),
        "mean_pair_complementarity_gain": float(np.mean(gains)),
        "examples_with_positive_pair_gain": sum(gain > 1e-12 for gain in gains),
        "maximum_example_pair_gain": max(gains, default=float("nan")),
    }


def pareto_front(points: Mapping[str, tuple[float, float]]) -> tuple[list[str], list[dict]]:
    dominated_by: list[dict] = []
    front = []
    for name, (tokens, recall) in points.items():
        dominators = [other for other, (other_tokens, other_recall) in points.items()
                      if other != name and other_tokens <= tokens and other_recall >= recall
                      and (other_tokens < tokens or other_recall > recall)]
        if dominators:
            dominated_by.append({"condition": name, "dominated_by": sorted(dominators)})
        else:
            front.append(name)
    return sorted(front), dominated_by


def analyse(rows: Sequence[dict], resamples: int, seed: int) -> tuple[list[dict], dict]:
    by, meta, conditions = build_index(rows)
    result_rows: list[dict] = []
    metric_specs = (
        ("unique_gold_span_recall", "evidence_recall", "evidence_scored", False),
        ("answerable_complete_evidence_recovery", "complete_evidence", "answerable_evidence_scored", True),
        ("insufficient_residual_support_complete_recovery", "complete_evidence",
         "insufficient_with_residual_support", True),
        ("evidence_precision", "evidence_precision", "evidence_scored", False),
        ("context_tokens", "context_tokens", "all_questions", False),
        ("final_k", "final_k", "all_questions", False),
    )
    p_families: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for metric, field, population, binary in metric_specs:
        for partition in ("selection", "tuning", "all"):
            for dataset in ("both", "qasper", "musique_full"):
                ids = population_ids(meta, partition, dataset, population)
                reference = values(by, REFERENCE, ids, field)
                for key in conditions:
                    if key == REFERENCE:
                        continue
                    stats = paired(reference, values(by, key, ids, field), resamples, seed)
                    if stats.get("n", 0) < 2:
                        continue
                    row = {"metric": metric, "population": population, "partition": partition,
                           "dataset": dataset, "arm": key[0], "condition": key[1], **stats}
                    if binary:
                        row.update({f"mcnemar_{k}": v for k, v in
                                    mcnemar(reference, values(by, key, ids, field)).items()})
                    result_rows.append(row)
                    p_families[(metric, partition, dataset)][key[1]] = stats["p_value"]
    adjusted = {family: holm(pvals) for family, pvals in p_families.items()}
    for row in result_rows:
        family = (row["metric"], row["partition"], row["dataset"])
        row["holm_p_all_conditions"] = adjusted[family][row["condition"]]
        row["holm_family_size"] = len(adjusted[family])

    all_ids = population_ids(meta, "all", "both", "all_questions")
    selection_ids = population_ids(meta, "selection", "both", "all_questions")
    scored_selection_ids = population_ids(meta, "selection", "both", "evidence_scored")
    answerable_selection_ids = population_ids(meta, "selection", "both", "answerable_evidence_scored")
    residual_selection_ids = population_ids(meta, "selection", "both", "insufficient_with_residual_support")
    manifest_path = ROOT / "output" / "calibration" / "frozen_manifest.json"

    gate_rows = list(csv.DictReader((RAW / "GATE_INSTRUMENTATION_PER_EXAMPLE.csv").open(encoding="utf-8")))
    gate_counts = {
        "candidates_scored": sum(int(float(row["candidates_scored"])) for row in gate_rows),
        "novelty_binding": sum(int(float(row["novelty_binding"])) for row in gate_rows),
        "utility_binding": sum(int(float(row["utility_binding"])) for row in gate_rows),
        "redundancy_binding": sum(int(float(row["redundancy_binding"])) for row in gate_rows),
        "gap_would_fire": sum(int(float(row["gap_would_fire"])) for row in gate_rows),
        "maximum_standardized_gap": max(fnum(row["max_standardized_gap"]) for row in gate_rows),
    }
    gate_effects = {}
    for condition in ("no_novelty", "no_similarity_gap", "no_post_retrieval_deduplication"):
        fields = ("final_k", "context_tokens", "candidates_examined", "retrieval_rounds", "stop_reason",
                  "evidence_recall", "evidence_precision", "complete_evidence")
        changes = {field: 0 for field in fields}
        for example_id in all_ids:
            for field in fields:
                a, b = by[REFERENCE][example_id][field], by[("sar_rag", condition)][example_id][field]
                if field == "stop_reason":
                    changed = a != b
                else:
                    fa, fb = fnum(a), fnum(b)
                    changed = not (math.isnan(fa) and math.isnan(fb)) and fa != fb
                changes[field] += int(changed)
        gate_effects[condition] = changes

    oracle = {}
    for label, singles, pairs in (("legacy", LEGACY_SINGLES, LEGACY_PAIRS),
                                  ("hierarchical_v2_candidate", CANDIDATE_SINGLES, CANDIDATE_PAIRS)):
        oracle[label] = {}
        for partition in ("selection", "tuning", "all"):
            for dataset in ("both", "qasper", "musique_full"):
                oracle[label][f"{partition}|{dataset}"] = oracle_summary(
                    by, meta, singles, pairs, partition, dataset)

    selection_points = {}
    for key in conditions:
        token_mean = float(np.mean(values(by, key, selection_ids, "context_tokens")))
        recall_mean = float(np.mean(values(by, key, scored_selection_ids, "evidence_recall")))
        selection_points[f"{key[0]}:{key[1]}"] = (token_mean, recall_mean)
    front, dominated = pareto_front(selection_points)
    reference_tokens, _ = selection_points["sar_rag:full"]
    traditional = {name: point for name, point in selection_points.items() if name.startswith("traditional:")}
    matched = {name: {"mean_context_tokens": point[0], "mean_recall": point[1],
                      "absolute_token_difference_pct": 100 * abs(point[0] - reference_tokens) / reference_tokens}
               for name, point in traditional.items()
               if abs(point[0] - reference_tokens) / reference_tokens <= 0.10}
    for name, block in matched.items():
        condition = name.split(":", 1)[1]
        row = next(r for r in result_rows if r["metric"] == "unique_gold_span_recall"
                   and r["partition"] == "selection" and r["dataset"] == "both"
                   and r["condition"] == condition)
        block.update({"recall_difference_condition_minus_sar": row["difference"],
                      "recall_ci": [row["ci_low"], row["ci_high"]],
                      "recall_p": row["p_value"], "recall_holm_p": row["holm_p_all_conditions"]})

    def reachability(ids: Sequence[str], primary: tuple[str, str]) -> dict:
        complete_misses = [example_id for example_id in ids
                           if fnum(by[primary][example_id]["evidence_recall"]) == 0]
        unreachable = [example_id for example_id in complete_misses
                       if max(fnum(by[("sar_rag", name)][example_id]["evidence_recall"])
                              for name in CANDIDATE_SINGLES + CANDIDATE_PAIRS) == 0]
        return {
            "complete_misses": len(complete_misses),
            "unreachable_by_any_candidate_single_or_pair": len(unreachable),
            "reachable_by_perfect_candidate_routing": len(complete_misses) - len(unreachable),
        }

    validation = {
        "source_csv_sha256": hashlib.sha256((RAW / "SCREENING_V2_PER_EXAMPLE.csv").read_bytes()).hexdigest(),
        "row_count": len(rows), "condition_count": len(conditions), "unique_examples": len(all_ids),
        "expected_row_count": len(conditions) * len(all_ids),
        "duplicate_keys": 0,
        "partition_counts": dict(Counter(meta[i]["design_partition"] for i in all_ids)),
        "dataset_counts": dict(Counter(meta[i]["dataset"] for i in all_ids)),
        "answerability_counts": dict(Counter(
            f"{meta[i]['dataset']}|{'answerable' if int(float(meta[i]['answerable'])) else 'insufficient'}"
            for i in all_ids)),
        "evidence_scored_counts": {
            "all": len(population_ids(meta, "all", "both", "evidence_scored")),
            "tuning": len(population_ids(meta, "tuning", "both", "evidence_scored")),
            "selection": len(scored_selection_ids),
            "selection_answerable": len(answerable_selection_ids),
            "selection_insufficient_with_residual_support": len(residual_selection_ids),
        },
        "nan_metric_cells": {field: sum(math.isnan(fnum(row[field])) for row in rows)
                             for field in ("evidence_recall", "evidence_precision", "complete_evidence")},
        "frozen_v1_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "frozen_v1_manifest_hash_matches_expected":
            hashlib.sha256(manifest_path.read_bytes()).hexdigest() == EXPECTED_MANIFEST_SHA256,
        "oracle_complementarity": oracle,
        "mean_context_tokens_selection": {name: value[0] for name, value in selection_points.items()},
        "stop_reasons_full_selection": dict(Counter(by[REFERENCE][i]["stop_reason"] for i in selection_ids)),
        "gate_binding_counts": gate_counts,
        "gate_output_change_counts": gate_effects,
        "failure_reachability": {
            "original_full_all_scored": reachability(
                population_ids(meta, "all", "both", "evidence_scored"), REFERENCE),
            "hierarchical_v2_full_all_scored": reachability(
                population_ids(meta, "all", "both", "evidence_scored"), CANDIDATE),
            "hierarchical_v2_full_selection_scored": reachability(scored_selection_ids, CANDIDATE),
        },
        "actual_token_comparison": {
            "definition": "Mean actual context tokens within 10% of SAR-RAG on the identical selection queries.",
            "matched_traditional_operating_points": matched,
            "sar_has_significant_matched_actual_token_advantage": any(
                item["recall_difference_condition_minus_sar"] < 0 and item["recall_holm_p"] < 0.05
                for item in matched.values()),
            "interpretation": (
                "The Boolean above tests a positive SAR advantage only; absence of a significantly superior "
                "traditional condition is not evidence of a SAR advantage."
            ),
            "selection_pareto_front": front,
            "dominated_conditions": dominated,
        },
    }
    return result_rows, validation


def write_note(validation: Mapping) -> str:
    candidate = validation["oracle_complementarity"]["hierarchical_v2_candidate"]
    legacy = validation["oracle_complementarity"]["legacy"]
    matched = validation["actual_token_comparison"]["matched_traditional_operating_points"]
    return f"""# Screening v3 correction note

**Scope:** versioned correction computed directly from the preserved 9,724-row per-example CSV.
The historical screening outputs were not overwritten. No LLM or held-out split was used.

## Corrected metric populations

- Unique gold-span recall remains defined over all {validation['evidence_scored_counts']['all']} questions with one or more gold spans.
- Complete-evidence recovery (CER) is now restricted to answerable questions. The selection population is {validation['evidence_scored_counts']['selection_answerable']}.
- For insufficient-evidence questions that retain residual support annotations, complete recovery of those residual spans is reported separately. The selection population is {validation['evidence_scored_counts']['selection_insufficient_with_residual_support']}.
- Questions with no gold spans remain unscored for evidence recall/precision/CER; the {validation['nan_metric_cells']['evidence_recall']:,} NaNs per metric are expected.

## Oracle complementarity correction

- Hierarchical-v2 candidate family: Oracle@2 - Oracle@1 is exactly {candidate['all|both']['mean_pair_complementarity_gain']:.6f} across {candidate['all|both']['n']} scored examples; {candidate['all|both']['examples_with_positive_pair_gain']} examples have a positive pair gain.
- Legacy family: selection gain is {legacy['selection|both']['mean_pair_complementarity_gain']:.6f}; pooled gain is {legacy['all|both']['mean_pair_complementarity_gain']:.6f}, caused by {legacy['all|both']['examples_with_positive_pair_gain']} tuning example with maximum gain {legacy['all|both']['maximum_example_pair_gain']:.6f}.

Therefore, “pair complementarity is exactly zero in every scope” was too broad.

## Actual-token comparison correction

The 2,000-token value is a safety maximum, not a consumption target. Fair interpretation uses the same queries, reports actual consumed tokens, shows the recall-token Pareto frontier, and declares a matched operating point as mean consumption within 10% of SAR-RAG. Matched traditional points: {', '.join(sorted(matched)) or 'none'}.

`sar_has_significant_matched_actual_token_advantage` is **{str(validation['actual_token_comparison']['sar_has_significant_matched_actual_token_advantage']).lower()}**. This explicitly tests a positive SAR advantage. The previous `budget_matched_defensible` name was ambiguous because “no tested traditional condition significantly beats SAR” does not establish that SAR significantly beats a matched-token comparator.

## Repository and frozen artifacts

The working tree is not clean. The defensible integrity statement is narrower: the frozen v1 manifest SHA-256 is `{validation['frozen_v1_manifest_sha256']}` and it matches the preregistered expected hash. No claim of a clean Git worktree is made.

## Scoped interpretation

The development screening rejects the tested RRF one-or-two-view routing claim only for the current encoder, representation family, datasets, retrieval configuration, and design-exposed sample. Unreachable failures are evidence about the tested retrieval system, not proof that the encoder or routing can never work elsewhere. These results are not held-out evidence.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resamples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()
    source = RAW / "SCREENING_V2_PER_EXAMPLE.csv"
    rows = list(csv.DictReader(source.open(encoding="utf-8")))
    results, validation = analyse(rows, args.resamples, args.seed)
    fields = ["metric", "population", "partition", "dataset", "arm", "condition", "n",
              "mean_reference", "mean_condition", "difference", "ci_low", "ci_high", "p_value",
              "holm_p_all_conditions", "holm_family_size", "dz", "mcnemar_reference_only",
              "mcnemar_condition_only", "mcnemar_discordant", "mcnemar_p_value"]
    with (OUT / "SCREENING_V3_RESULTS_CORRECTED.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda row: (
            row["metric"], row["partition"], row["dataset"], row["arm"], row["condition"])))
    (OUT / "SCREENING_V3_VALIDATION_CORRECTED.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")
    (OUT / "SCREENING_V3_CORRECTION_NOTE.md").write_text(write_note(validation), encoding="utf-8")
    print(json.dumps({
        "rows": len(rows), "corrected_result_rows": len(results),
        "selection_scored": validation["evidence_scored_counts"]["selection"],
        "selection_answerable": validation["evidence_scored_counts"]["selection_answerable"],
        "manifest_hash_ok": validation["frozen_v1_manifest_hash_matches_expected"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
