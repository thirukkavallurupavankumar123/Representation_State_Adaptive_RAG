"""Versioned post-analysis for the completed v2 retrieval-only recalibration."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "output" / "adaptive_single_view_v2" / "recalibration_seed20260905"
RESAMPLES = 20_000
SEED = 20260905


def paired(reference: list[float], condition: list[float]) -> dict:
    pairs = [(a, b) for a, b in zip(reference, condition)
             if not math.isnan(a) and not math.isnan(b)]
    delta = np.asarray([b - a for a, b in pairs], dtype=float)
    rng = np.random.default_rng(SEED)
    bootstrap = delta[rng.integers(0, len(delta), size=(RESAMPLES, len(delta)))].mean(axis=1)
    signs = rng.integers(0, 2, size=(RESAMPLES, len(delta))) * 2 - 1
    randomised = (delta * signs).mean(axis=1)
    observed = float(delta.mean())
    return {"n": len(delta), "mean_reference": float(np.mean([a for a, _ in pairs])),
            "mean_selected": float(np.mean([b for _, b in pairs])), "difference": observed,
            "ci_low": float(np.quantile(bootstrap, .025)),
            "ci_high": float(np.quantile(bootstrap, .975)),
            "p_value": float((np.sum(np.abs(randomised) >= abs(observed)) + 1) / (RESAMPLES + 1))}


def holm(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: item[1])
    output, running = {}, 0.0
    for rank, (name, value) in enumerate(ordered, 1):
        running = max(running, min(1.0, (len(ordered) - rank + 1) * value))
        output[name] = running
    return output


def pareto(aggregates: list[dict]) -> tuple[list[str], list[dict]]:
    front, dominated = [], []
    for row in aggregates:
        name = row["configuration_id"]
        tokens, recall = float(row["context_tokens"]), float(row["evidence_recall"])
        dominators = [other["configuration_id"] for other in aggregates
                      if other["configuration_id"] != name
                      and float(other["context_tokens"]) <= tokens
                      and float(other["evidence_recall"]) >= recall
                      and (float(other["context_tokens"]) < tokens
                           or float(other["evidence_recall"]) > recall)]
        if dominators:
            dominated.append({"configuration_id": name, "dominated_by": sorted(dominators)})
        else:
            front.append(name)
    return sorted(front), dominated


def main() -> int:
    manifest = json.loads((RUN / "FROZEN_V2_MANIFEST.json").read_text(encoding="utf-8"))
    winner_id = manifest["selected_configuration_id"]
    with (RUN / "SELECTION_PER_EXAMPLE.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with (RUN / "SELECTION_AGGREGATE.csv").open(encoding="utf-8", newline="") as stream:
        aggregates = list(csv.DictReader(stream))
    winner_rows = {row["example_id"]: row for row in rows if row["configuration_id"] == winner_id}
    winner_aggregate = next(row for row in aggregates if row["configuration_id"] == winner_id)
    selected_tokens = float(winner_aggregate["context_tokens"])

    matched_aggregates = [row for row in aggregates if row["configuration_id"] != winner_id
                          and abs(float(row["context_tokens"]) - selected_tokens) / selected_tokens <= 0.10]
    comparisons = []
    families: dict[str, dict[str, float]] = defaultdict(dict)
    for comparator in matched_aggregates:
        comparator_rows = {row["example_id"]: row for row in rows
                           if row["configuration_id"] == comparator["configuration_id"]}
        ids = sorted(set(winner_rows) & set(comparator_rows))
        for metric, field, answerable_only in (
            ("evidence_recall", "evidence_recall", False),
            ("answerable_CER", "complete_evidence", True),
            ("context_tokens", "context_tokens", False),
        ):
            usable = [example_id for example_id in ids
                      if not answerable_only or int(winner_rows[example_id]["answerable"]) == 1]
            stats = paired([float(comparator_rows[i][field]) for i in usable],
                           [float(winner_rows[i][field]) for i in usable])
            comparison = {
                "metric": metric, "reference_configuration_id": comparator["configuration_id"],
                "reference_mode": comparator["mode"],
                "reference_representation": comparator["representation"],
                "reference_token_point": comparator["token_point"],
                "reference_mean_actual_tokens": comparator["context_tokens"],
                "actual_token_difference_pct":
                    100 * (selected_tokens - float(comparator["context_tokens"])) / float(comparator["context_tokens"]),
                **stats,
            }
            comparisons.append(comparison)
            families[metric][comparator["configuration_id"]] = stats["p_value"]
    adjusted = {metric: holm(values) for metric, values in families.items()}
    for row in comparisons:
        row["holm_p_matched_family"] = adjusted[row["metric"]][row["reference_configuration_id"]]
        row["holm_family_size"] = len(adjusted[row["metric"]])
    fields = list(comparisons[0]) if comparisons else ["metric"]
    with (RUN / "MATCHED_ACTUAL_TOKEN_COMPARISONS.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(comparisons)

    front, dominated = pareto(aggregates)
    primary_aggregates = [row for row in aggregates if row["mode"] != "v1_full"]
    primary_front, primary_dominated = pareto(primary_aggregates)
    primary = list(csv.DictReader((RUN / "PAIRED_UNCERTAINTY.csv").open(encoding="utf-8", newline="")))
    lookup = {(row["metric"], row["reference"]): row for row in primary}
    fixed20_recall = lookup[("evidence_recall", "fixed_k20")]
    fixed20_tokens = lookup[("context_tokens", "fixed_k20")]
    reduction = -100 * float(fixed20_tokens["difference"]) / float(fixed20_tokens["mean_reference"])
    analysis = {
        "status": "DEVELOPMENT_SELECTION_ONLY_NOT_HELD_OUT",
        "selected_configuration_id": winner_id,
        "selected_metrics": {key: winner_aggregate[key] for key in (
            "representation", "token_point", "evidence_recall", "answerable_CER",
            "insufficient_residual_support_complete_recovery", "context_tokens", "final_k",
            "candidates_examined", "retrieval_rounds")},
        "matched_actual_token_rule": "mean actual tokens within 10% of selected v2 on identical 63 selection queries",
        "matched_configuration_count": len(matched_aggregates),
        "pareto_front_configuration_ids": front,
        "selected_is_pareto_nondominated": winner_id in front,
        "dominated_conditions": dominated,
        "primary_v2_and_fixed_comparator_pareto_front": primary_front,
        "selected_is_nondominated_excluding_descriptive_v1": winner_id in primary_front,
        "primary_v2_and_fixed_comparator_dominated_conditions": primary_dominated,
        "scope_specific_conclusions": {
            "quality_vs_probe": {
                "recall_gain": float(lookup[("evidence_recall", "probe_only")]["difference"]),
                "holm_p": float(lookup[("evidence_recall", "probe_only")]["holm_p"]),
                "supported": float(lookup[("evidence_recall", "probe_only")]["difference"]) >= .02
                             and float(lookup[("evidence_recall", "probe_only")]["holm_p"]) < .05,
            },
            "quality_vs_same_view_k10": {
                "recall_gain": float(lookup[("evidence_recall", "fixed_k10")]["difference"]),
                "context_increase": float(lookup[("context_tokens", "fixed_k10")]["difference"]),
                "holm_p_recall": float(lookup[("evidence_recall", "fixed_k10")]["holm_p"]),
                "supported_with_more_context": float(lookup[("evidence_recall", "fixed_k10")]["difference"]) >= .02
                    and float(lookup[("evidence_recall", "fixed_k10")]["holm_p"]) < .05,
            },
            "efficiency_vs_same_view_k20": {
                "recall_difference": float(fixed20_recall["difference"]),
                "recall_ci": [float(fixed20_recall["ci_low"]), float(fixed20_recall["ci_high"])],
                "context_reduction_percent": reduction,
                "holm_p_tokens": float(fixed20_tokens["holm_p"]),
                "supported_under_noninferiority": float(fixed20_recall["ci_low"]) >= -.02
                    and reduction >= 10 and float(fixed20_tokens["holm_p"]) < .05,
            },
        },
        "stop_reasons": dict(Counter(row["stop_reason"] for row in winner_rows.values())),
        "final_k_range": [min(int(row["final_k"]) for row in winner_rows.values()),
                          max(int(row["final_k"]) for row in winner_rows.values())],
        "failure_counts": {
            "total_miss": sum(float(row["evidence_recall"]) == 0 for row in winner_rows.values()),
            "partial": sum(0 < float(row["evidence_recall"]) < 1 for row in winner_rows.values()),
            "complete": sum(float(row["evidence_recall"]) == 1 for row in winner_rows.values()),
        },
        "overall_classification": "unsupported",
        "overall_classification_reason": (
            "Frozen v1 point-dominates selected v2 on mean recall and mean actual tokens. The paired "
            "differences are not statistically significant and the nominal maxima differ, so this is not "
            "evidence that v1 is superior; it is sufficient to block a v2 replacement/superiority claim."
        ),
        "paper_safe_interpretation": (
            "On a fresh but development-exposed training selection sample, the frozen v2 rule chose paragraph "
            "retrieval at the 1,600-token operating point. It improves retrieval over the shallow probe and "
            "same-view k=10, but the k=10 comparison uses less context. Relative to same-view k=20, v2 is "
            "recall-noninferior within the preregistered margin and uses less context. However, frozen v1 has "
            "slightly higher recall and lower actual tokens on point estimates, so the v2 replacement claim is "
            "unsupported. The scoped comparator results are developmental, not held-out answer-quality evidence."
        ),
    }
    (RUN / "PARETO_AND_DECISION.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    (RUN / "RESEARCH_DECISION.md").write_text(f"""# Adaptive Single-View V2 recalibration decision

**Status:** development selection only; retrieval-only; zero LLM calls; not held-out evidence.

The frozen rule selected **paragraph** retrieval at the **1,600-token operating point** with relevance percentile 0.45, patience 1, six retrieval rounds, 30 examined candidates, and a 20-chunk cap. On 63 fresh selection questions it reached recall {float(winner_aggregate['evidence_recall']):.4f}, answerable-only CER {float(winner_aggregate['answerable_CER']):.4f}, and mean actual context {selected_tokens:.1f} tokens. Final k was emergent, ranging from {analysis['final_k_range'][0]} to {analysis['final_k_range'][1]} (mean {float(winner_aggregate['final_k']):.2f}).

Against the same-view shallow probe, recall increased by {analysis['scope_specific_conclusions']['quality_vs_probe']['recall_gain']:+.4f} (Holm p={analysis['scope_specific_conclusions']['quality_vs_probe']['holm_p']:.4g}). Against same-view k=10, recall increased by {analysis['scope_specific_conclusions']['quality_vs_same_view_k10']['recall_gain']:+.4f}, but v2 used {analysis['scope_specific_conclusions']['quality_vs_same_view_k10']['context_increase']:.1f} more tokens. Against same-view k=20, recall differed by {analysis['scope_specific_conclusions']['efficiency_vs_same_view_k20']['recall_difference']:+.4f}, CI [{analysis['scope_specific_conclusions']['efficiency_vs_same_view_k20']['recall_ci'][0]:+.4f}, {analysis['scope_specific_conclusions']['efficiency_vs_same_view_k20']['recall_ci'][1]:+.4f}], while v2 reduced context by {analysis['scope_specific_conclusions']['efficiency_vs_same_view_k20']['context_reduction_percent']:.1f}%.

The selected point is {'non-dominated' if analysis['selected_is_pareto_nondominated'] else 'dominated'} among all tested selection operating points. Frozen v1 is the sole point-estimate dominator: recall 0.7460 versus 0.7381 and actual context 1,147.6 versus 1,199.2 tokens. The paired recall difference (v2 minus v1) is -0.0079, CI [-0.0675, +0.0476], and the token difference is +51.6, CI [-2.9, +100.3]; neither is significant after correction. V1 also had a larger nominal safety maximum, so this is not evidence that v1 is superior, but it blocks a v2 replacement claim.

The required overall category is therefore **unsupported**. Scoped findings remain: v2 retrieves better than a shallow probe and k=10, and is more efficient than same-view k=20 under recall noninferiority. It does not establish held-out answer quality, universal superiority, or a benefit from representation routing.
""", encoding="utf-8")
    print(json.dumps({"matched": len(matched_aggregates), "pareto": winner_id in front,
                      "scope_specific": analysis["scope_specific_conclusions"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
