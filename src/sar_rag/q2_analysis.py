from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np

from .ablations import registry
from .experiments import (
    classification_summary, exact_mcnemar_p_value, holm_adjust, paired_statistics,
)


PRIMARY_METRICS = (
    "answer_token_f1", "evidence_set_recall", "complete_evidence_recovery", "context_tokens"
)
QUALITY_METRICS = (
    "answer_exact_match", "answer_token_precision", "answer_token_recall", "answer_token_f1",
    "evidence_set_precision", "evidence_set_recall", "complete_evidence_recovery",
    "retrieval_mrr", "retrieval_ndcg_at_5", "retrieval_ndcg_at_10",
    "answerability_correct", "faithfulness", "answer_relevancy",
)
EFFICIENCY_METRICS = (
    "latency_seconds", "context_tokens", "prompt_tokens", "completion_tokens", "total_tokens",
    "estimated_cost_usd", "final_k", "candidates_examined", "retrieval_rounds",
)


def _clean(values):
    return [float(value) for value in values if value is not None and not math.isnan(float(value))]


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def descriptive(values) -> dict:
    values = _clean(values)
    if not values:
        return {"n": 0, "mean": None, "median": None, "sd": None, "p25": None,
                "p75": None, "p95": None, "min": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "n": len(values), "mean": float(array.mean()), "median": float(np.median(array)),
        "sd": float(array.std(ddof=1)) if len(values) > 1 else 0.0,
        "p25": float(np.quantile(array, .25)), "p75": float(np.quantile(array, .75)),
        "p95": float(np.quantile(array, .95)), "min": float(array.min()),
        "max": float(array.max()),
    }


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = (start + stop - 1) / 2 + 1
        for position in range(start, stop):
            ranks[order[position]] = rank
        start = stop
    return ranks


def correlation(x, y) -> dict:
    pairs = [(float(a), float(b)) for a, b in zip(x, y)
             if a is not None and b is not None and not math.isnan(float(a)) and not math.isnan(float(b))]
    if len(pairs) < 3:
        return {"n": len(pairs), "pearson_r": None, "spearman_rho": None}
    left, right = zip(*pairs)
    if np.std(left) == 0 or np.std(right) == 0:
        pearson = None
    else:
        pearson = float(np.corrcoef(left, right)[0, 1])
    ranks_left, ranks_right = _rank(list(left)), _rank(list(right))
    spearman = None if np.std(ranks_left) == 0 or np.std(ranks_right) == 0 else float(
        np.corrcoef(ranks_left, ranks_right)[0, 1]
    )
    return {"n": len(pairs), "pearson_r": pearson, "spearman_rho": spearman}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_results(paths: list[Path]) -> tuple[list[dict], list[dict]]:
    records, summaries = [], []
    seen = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload["summary"]
        summaries.append({"path": str(path), **summary})
        condition = summary.get("registered_condition", "full")
        for record in payload["records"]:
            key = (condition, record["example_id"])
            if key in seen:
                raise ValueError(f"duplicate condition/example pair: {key}")
            seen.add(key)
            records.append({"condition": condition, **record})
    return records, summaries


def dataset_statistics(paths: list[Path]) -> list[dict]:
    output = []
    for path in paths:
        rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        by_dataset = defaultdict(list)
        for row in rows:
            by_dataset[(row.get("dataset"), row.get("split"))].append(row)
        for (dataset, split), group in by_dataset.items():
            output.append({
                "dataset": dataset, "split": split, "questions": len(group),
                "unique_ids": len({row["example_id"] for row in group}),
                "answerable": sum(row.get("gold_answerability") == "ANSWERABLE" for row in group),
                "insufficient": sum(row.get("gold_answerability") == "INSUFFICIENT" for row in group),
                "documents_mean": mean(len(row["documents"]) for row in group),
                "document_words_mean": mean(
                    sum(len(document["text"].split()) for document in row["documents"])
                    for row in group
                ),
                "gold_evidence_spans_mean": mean(len(row.get("gold_evidence_spans", [])) for row in group),
            })
    return output


def main_results(records: list[dict], resamples: int, seed: int) -> tuple[list[dict], dict]:
    rows, details = [], {}
    full = [record for record in records if record["condition"] == "full"]
    for dataset in sorted({record.get("dataset") for record in full}):
        group = [record for record in full if record.get("dataset") == dataset]
        statistics = paired_statistics(group, resamples, seed)
        primary_p = {name: statistics[name]["paired_randomization_p_value"]
                     for name in PRIMARY_METRICS if name in statistics}
        adjusted = holm_adjust(primary_p)
        mcnemar = {}
        for metric in ("answer_exact_match", "complete_evidence_recovery", "answerability_correct"):
            pairs = [(record["traditional"].get(metric), record["proposed"].get(metric)) for record in group]
            pairs = [(a, b) for a, b in pairs if a is not None and b is not None
                     and not math.isnan(float(a)) and not math.isnan(float(b))]
            if pairs:
                mcnemar[metric] = exact_mcnemar_p_value(
                    [int(a) for a, _ in pairs], [int(b) for _, b in pairs]
                )
        details[dataset] = {"paired_statistics": statistics, "holm": adjusted, "mcnemar": mcnemar}
        for metric, result in statistics.items():
            if metric not in set(QUALITY_METRICS) | set(EFFICIENCY_METRICS):
                continue
            rows.append({
                "dataset": dataset, "metric": metric,
                "traditional": result["traditional_mean"], "sar_rag": result["proposed_mean"],
                "difference": result["paired_difference_proposed_minus_traditional"],
                "ci_low": result["bootstrap_95_ci"][0], "ci_high": result["bootstrap_95_ci"][1],
                "randomization_p": result["paired_randomization_p_value"],
                "holm_p": adjusted.get(metric), "effect_size_dz": result["paired_effect_size_dz"],
                "paired_n": result["paired_n"],
                "mcnemar_p": mcnemar.get(metric, {}).get("exact_mcnemar_p_value"),
            })
    return rows, details


def efficiency(records: list[dict]) -> list[dict]:
    output = []
    full = [record for record in records if record["condition"] == "full"]
    for dataset in sorted({record.get("dataset") for record in full}):
        group = [record for record in full if record.get("dataset") == dataset]
        for method in ("traditional", "proposed"):
            for metric in EFFICIENCY_METRICS:
                stats = descriptive(record[method].get(metric) for record in group)
                output.append({"dataset": dataset, "method": method, "metric": metric, **stats})
    return output


def representation_analysis(records: list[dict]) -> tuple[list[dict], dict]:
    groups = defaultdict(list)
    full = [record for record in records if record["condition"] == "full"]
    for record in full:
        state = "+".join(sorted(record.get("selected_representations", []))) or "none"
        groups[(record.get("dataset"), state)].append(record)
    rows = []
    for (dataset, state), group in sorted(groups.items()):
        rows.append({
            "dataset": dataset, "state": state, "questions": len(group),
            "percentage": 100 * len(group) / sum(r.get("dataset") == dataset for r in full),
            "representations": len(state.split("+")) if state != "none" else 0,
            "answer_f1": descriptive(r["proposed"].get("answer_token_f1") for r in group)["mean"],
            "evidence_recall": descriptive(r["proposed"].get("evidence_set_recall") for r in group)["mean"],
            "complete_evidence": descriptive(r["proposed"].get("complete_evidence_recovery") for r in group)["mean"],
            "context_tokens": descriptive(r["proposed"].get("context_tokens") for r in group)["mean"],
            "final_k": descriptive(r["proposed"].get("final_k") for r in group)["mean"],
        })
    individual = Counter(rep for record in full for rep in record.get("selected_representations", []))
    second_gains = []
    routing_risks = 0
    second_representation_unique_gold_cases = 0
    for record in full:
        profiles = record.get("trace_summary", {}).get("proposed", {}).get("representation_profiles", {})
        selected = set(record.get("selected_representations", []))
        selected_profiles = sorted(
            (value.get("marginal_gain", 0) for name, value in profiles.items() if name in selected)
        )
        if len(selected_profiles) > 1:
            second_gains.append(selected_profiles[0])
        diagnostics = record.get("representation_gold_diagnostic", {})
        if len(selected) == 2 and profiles:
            ordered_selected = sorted(
                selected, key=lambda name: profiles.get(name, {}).get("marginal_gain", 0), reverse=True
            )
            first_gold = set(diagnostics.get(ordered_selected[0], {}).get("gold_span_indices_in_top_probe", []))
            second_gold = set(diagnostics.get(ordered_selected[1], {}).get("gold_span_indices_in_top_probe", []))
            if second_gold - first_gold:
                second_representation_unique_gold_cases += 1
        if any(value.get("gold_span_count_in_top_probe", 0) for name, value in diagnostics.items()
               if name not in selected) and not any(
                   value.get("gold_span_count_in_top_probe", 0) for name, value in diagnostics.items()
                   if name in selected):
            routing_risks += 1
    summary = {
        "individual_selection_counts": dict(individual),
        "one_representation_percentage": 100 * sum(len(r.get("selected_representations", [])) == 1 for r in full) / len(full) if full else None,
        "two_representation_percentage": 100 * sum(len(r.get("selected_representations", [])) == 2 for r in full) / len(full) if full else None,
        "second_representation_marginal_gain": descriptive(second_gains),
        "routing_risk_cases_unselected_probe_has_gold_but_selected_probe_does_not": routing_risks,
        "second_representation_adds_unique_probe_gold_cases": second_representation_unique_gold_cases,
    }
    return rows, summary


def adaptive_k_analysis(records: list[dict], cap: int = 20, token_cap: int = 2000) -> dict:
    full = [record for record in records if record["condition"] == "full"]
    values = [record["proposed"].get("final_k") for record in full]
    stop_reasons = Counter(record.get("stop_reason", {}).get("proposed") for record in full)
    pause_reasons = Counter()
    for record in full:
        pause_reasons.update(
            record.get("trace_summary", {}).get("proposed", {}).get("pause_reasons", {})
        )
    return {
        "final_k": descriptive(values),
        "percentage_at_chunk_cap": 100 * sum(value == cap for value in values) / len(values) if values else None,
        "percentage_at_token_cap": 100 * sum(record["proposed"].get("context_tokens", 0) >= token_cap for record in full) / len(full) if full else None,
        "stop_reason_counts": dict(stop_reasons),
        "representation_pause_reason_counts": dict(pause_reasons),
        "correlation_final_k_gold_evidence_count": correlation(
            values, [record.get("gold_evidence_count", record.get("descriptors", {}).get("gold_evidence_count")) for record in full]
        ),
        "correlation_final_k_hop_count": correlation(
            values, [record.get("descriptors", {}).get("hop_count") for record in full]
        ),
    }


def subgroup_analysis(records: list[dict], resamples: int, seed: int) -> list[dict]:
    full = [record for record in records if record["condition"] == "full"]
    dimensions = {
        "dataset": lambda r: r.get("dataset"),
        "answerability": lambda r: r.get("gold_answerability"),
        "document_length": lambda r: r.get("descriptors", {}).get("document_length_group"),
        "hop_count": lambda r: r.get("descriptors", {}).get("hop_count"),
        "answer_length": lambda r: r.get("descriptors", {}).get("answer_length_group"),
        "gold_evidence_count": lambda r: r.get("gold_evidence_count", r.get("descriptors", {}).get("gold_evidence_count")),
        "document_count": lambda r: r.get("document_count", r.get("descriptors", {}).get("document_count")),
        "stop_reason": lambda r: r.get("stop_reason", {}).get("proposed"),
        "selected_representation_count": lambda r: len(r.get("selected_representations", [])),
    }
    output = []
    for dimension, getter in dimensions.items():
        groups = defaultdict(list)
        for record in full:
            value = getter(record)
            if value is not None:
                groups[str(value)].append(record)
        for value, group in sorted(groups.items()):
            paired = paired_statistics(group, resamples, seed)
            for metric in ("answer_token_f1", "evidence_set_recall",
                           "complete_evidence_recovery", "context_tokens"):
                result = paired.get(metric)
                if result is None:
                    continue
                output.append({
                    "dimension": dimension, "group": value, "metric": metric,
                    "traditional": result["traditional_mean"], "sar_rag": result["proposed_mean"],
                    "difference": result["paired_difference_proposed_minus_traditional"],
                    "ci_low": result["bootstrap_95_ci"][0], "ci_high": result["bootstrap_95_ci"][1],
                    "p_value": result["paired_randomization_p_value"], "paired_n": result["paired_n"],
                })
    return output


def quality_efficiency_reductions(records: list[dict]) -> list[dict]:
    output = []
    full = [record for record in records if record["condition"] == "full"]
    for dataset in sorted({record.get("dataset") for record in full}):
        group = [record for record in full if record.get("dataset") == dataset]
        for metric in ("context_tokens", "total_tokens", "estimated_cost_usd", "latency_seconds"):
            traditional = descriptive(record["traditional"].get(metric) for record in group)["mean"]
            proposed = descriptive(record["proposed"].get(metric) for record in group)["mean"]
            output.append({
                "dataset": dataset, "metric": metric, "traditional": traditional,
                "sar_rag": proposed, "reduction_percent": (
                    100 * (1 - proposed / traditional)
                    if traditional not in (None, 0) and proposed is not None else None
                ),
            })
    return output


def failure_analysis(records: list[dict], sample_size: int) -> tuple[list[dict], list[dict]]:
    categories, candidates = Counter(), []
    for record in (r for r in records if r["condition"] == "full"):
        proposed, traditional = record["proposed"], record["traditional"]
        labels = []
        if record.get("gold_answerability") == "INSUFFICIENT" and proposed.get("predicted_answerable") == 1:
            labels.append("false_answer_on_insufficient")
        if record.get("gold_answerability") == "INSUFFICIENT" and proposed.get("predicted_answerable") == 0:
            labels.append("correct_abstention")
        if record.get("gold_answerability") == "ANSWERABLE" and proposed.get("predicted_answerable") == 0:
            labels.append("incorrect_abstention")
        p_recall, t_recall = proposed.get("evidence_set_recall"), traditional.get("evidence_set_recall")
        if p_recall is not None and t_recall is not None and not math.isnan(float(p_recall)) and p_recall < t_recall:
            labels.append("sar_missed_evidence_recovered_by_baseline")
        if p_recall is not None and not math.isnan(float(p_recall)) and p_recall >= .999 and proposed.get("answer_token_f1", 1) < .5:
            labels.append("evidence_retrieved_but_answer_low_f1")
        gold = [" ".join(str(span).casefold().split()) for span in record.get("gold_evidence_spans", [])]
        provisional = " ".join(
            item.get("text", "") for item in record.get("provisional_evidence", {}).get("proposed", [])
        ).casefold()
        final = " ".join(
            item.get("text", "") for item in record.get("evidence", {}).get("proposed", [])
        ).casefold()
        if any(span and span in provisional and span not in final for span in gold):
            labels.append("gold_evidence_removed_after_retrieval")
        diagnostics = record.get("representation_gold_diagnostic", {})
        selected = set(record.get("selected_representations", []))
        selected_coverage = set().union(*(
            set(value.get("gold_span_indices_in_top_probe", []))
            for name, value in diagnostics.items() if name in selected
        )) if selected else set()
        unselected_coverage = set().union(*(
            set(value.get("gold_span_indices_in_top_probe", []))
            for name, value in diagnostics.items() if name not in selected
        )) if diagnostics else set()
        if unselected_coverage - selected_coverage:
            labels.append("unselected_representation_probe_has_unique_gold")
        if proposed.get("answer_token_f1", 0) + .2 < traditional.get("answer_token_f1", 0):
            labels.append("large_answer_f1_regression")
        if labels:
            categories.update(labels)
            candidates.append({
                "example_id": record["example_id"], "dataset": record.get("dataset"),
                "categories": labels, "query": record.get("query"),
                "traditional_answer": record.get("answers", {}).get("traditional"),
                "sar_answer": record.get("answers", {}).get("proposed"),
                "traditional_evidence_recall": t_recall, "sar_evidence_recall": p_recall,
                "manual_review_status": "not_reviewed",
            })
    candidates.sort(key=lambda item: sha256(item["example_id"].encode()).hexdigest())
    return ([{"category": key, "count": value} for key, value in categories.most_common()],
            candidates[:sample_size])


def ablation_table(records: list[dict]) -> list[dict]:
    output = []
    for condition in sorted({record["condition"] for record in records}):
        group = [record for record in records if record["condition"] == condition]
        for dataset in sorted({record.get("dataset") for record in group}):
            subset = [record for record in group if record.get("dataset") == dataset]
            method = "traditional" if condition.startswith("baseline_fixed_k_") else "proposed"
            output.append({
                "condition": condition, "dataset": dataset, "n": len(subset),
                "reported_method": method,
                "answer_f1": descriptive(r[method].get("answer_token_f1") for r in subset)["mean"],
                "evidence_recall": descriptive(r[method].get("evidence_set_recall") for r in subset)["mean"],
                "complete_evidence": descriptive(r[method].get("complete_evidence_recovery") for r in subset)["mean"],
                "context_tokens": descriptive(r[method].get("context_tokens") for r in subset)["mean"],
                "total_tokens": descriptive(r[method].get("total_tokens") for r in subset)["mean"],
                "api_calls": descriptive(r[method].get("api_calls") for r in subset)["mean"],
            })
    return output


def make_figures(records: list[dict], output: Path) -> list[str]:
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    full = [record for record in records if record["condition"] == "full"]
    figures = []
    if not full:
        return figures
    datasets = sorted({record.get("dataset") for record in full})

    def save(name):
        path = output / name
        plt.tight_layout(); plt.savefig(path, dpi=180); plt.close(); figures.append(str(path))

    metrics = [
        metric for metric in ("answer_token_f1", "evidence_set_recall", "complete_evidence_recovery")
        if descriptive(record["traditional"].get(metric) for record in full)["n"]
        and descriptive(record["proposed"].get(metric) for record in full)["n"]
    ]
    x = np.arange(len(metrics)); width = .36
    if metrics:
        plt.figure(figsize=(8, 4.5))
        plt.bar(x - width/2, [descriptive(r["traditional"].get(m) for r in full)["mean"] for m in metrics], width, label="Traditional")
        plt.bar(x + width/2, [descriptive(r["proposed"].get(m) for r in full)["mean"] for m in metrics], width, label="SAR-RAG")
        plt.xticks(x, [m.replace("_", "\n") for m in metrics]); plt.ylim(0, 1); plt.legend(); plt.title("Primary quality outcomes")
        save("01_main_quality.png")

    plt.figure(figsize=(7, 5))
    for method, marker in (("traditional", "o"), ("proposed", "^")):
        plt.scatter([r[method].get("context_tokens") for r in full], [r[method].get("evidence_set_recall") for r in full], alpha=.28, s=15, marker=marker, label=method)
    plt.xlabel("Context tokens"); plt.ylabel("Gold-evidence recall"); plt.title("Quality-efficiency observations"); plt.legend()
    save("02_recall_token_pareto.png")

    plt.figure(figsize=(7, 4.5)); plt.hist([r["proposed"].get("final_k", 0) for r in full], bins=np.arange(-.5, 21.5, 1), edgecolor="white")
    plt.xlabel("Emergent final k"); plt.ylabel("Questions"); plt.title("SAR-RAG final-k distribution")
    save("03_final_k.png")

    counts = Counter(rep for r in full for rep in r.get("selected_representations", []))
    plt.figure(figsize=(7, 4.5)); plt.bar(counts.keys(), counts.values()); plt.xticks(rotation=20, ha="right"); plt.ylabel("Selections"); plt.title("Representation selection frequency")
    save("04_representation_frequency.png")

    hop_groups = sorted({r.get("descriptors", {}).get("hop_count") for r in full if r.get("descriptors", {}).get("hop_count") is not None})
    if hop_groups:
        plt.figure(figsize=(7, 4.5)); x = np.arange(len(hop_groups))
        for offset, method in ((-.18, "traditional"), (.18, "proposed")):
            plt.bar(x + offset, [descriptive(r[method].get("evidence_set_recall") for r in full if r.get("descriptors", {}).get("hop_count") == hop)["mean"] for hop in hop_groups], .36, label=method)
        plt.xticks(x, hop_groups); plt.xlabel("Hop count"); plt.ylabel("Evidence recall"); plt.legend(); plt.title("Retrieval by hop count")
        save("05_hop_robustness.png")

    plt.figure(figsize=(8, 4.5)); labels = ["Latency", "Context tokens", "Total tokens"]
    ratios = []
    for metric in ("latency_seconds", "context_tokens", "total_tokens"):
        t = descriptive(r["traditional"].get(metric) for r in full)["mean"] or 0
        p = descriptive(r["proposed"].get(metric) for r in full)["mean"] or 0
        ratios.append(100 * p / t if t else 0)
    plt.bar(labels, ratios); plt.axhline(100, color="black", linestyle="--", linewidth=1); plt.ylabel("SAR as % of Traditional"); plt.title("Efficiency ratio (lower is better)")
    save("06_efficiency.png")

    ablations = ablation_table(records)
    if len({row["condition"] for row in ablations}) > 1:
        aggregate = defaultdict(list)
        for row in ablations: aggregate[row["condition"]].append(row["evidence_recall"])
        names = list(aggregate); values = [descriptive(aggregate[name])["mean"] for name in names]
        plt.figure(figsize=(10, 5)); plt.bar(names, values); plt.xticks(rotation=55, ha="right"); plt.ylabel("Evidence recall"); plt.title("Registered ablations")
        save("07_ablations.png")

    failures, _ = failure_analysis(records, 0)
    if failures:
        plt.figure(figsize=(9, 5)); plt.barh([row["category"] for row in failures], [row["count"] for row in failures]); plt.xlabel("Cases"); plt.title("Automatically flagged failure categories")
        save("08_failures.png")
    return figures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Q2-paper analyses from reproduced result files")
    parser.add_argument("--results", type=Path, action="append", required=True)
    parser.add_argument("--dataset-input", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--failure-sample", type=int, default=100)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, run_summaries = load_results(args.results)
    main_rows, inference = main_results(records, args.resamples, args.seed)
    efficiency_rows = efficiency(records)
    representation_rows, representation_summary = representation_analysis(records)
    adaptive = adaptive_k_analysis(records)
    subgroup_rows = subgroup_analysis(records, args.resamples, args.seed)
    reductions = quality_efficiency_reductions(records)
    failure_rows, failure_sample = failure_analysis(records, args.failure_sample)
    ablation_rows = ablation_table(records)
    datasets = dataset_statistics(args.dataset_input)
    figures = make_figures(records, args.output_dir / "figures")
    payload = {
        "status": "confirmatory_analysis" if all(s.get("status") == "confirmatory" for s in run_summaries) else "pilot_or_incomplete_not_for_claims",
        "run_summaries": run_summaries, "dataset_statistics": datasets,
        "main_results": main_rows, "inference": inference,
        "efficiency": efficiency_rows, "representation_states": representation_rows,
        "representation_summary": representation_summary, "adaptive_k": adaptive,
        "subgroups": subgroup_rows, "quality_efficiency_reductions": reductions,
        "failure_categories": failure_rows,
        "failure_review_sample": failure_sample, "ablation_registry": registry(),
        "ablation_results": ablation_rows, "figures": figures,
        "interpretation_rule": "No superiority claim unless confirmatory status, corrected primary inference, practical margins, and failure review all support it.",
    }
    (args.output_dir / "q2_analysis.json").write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    _write_csv(args.output_dir / "table_main_results.csv", main_rows)
    _write_csv(args.output_dir / "table_dataset_statistics.csv", datasets)
    _write_csv(args.output_dir / "table_efficiency.csv", efficiency_rows)
    _write_csv(args.output_dir / "table_representation_states.csv", representation_rows)
    _write_csv(args.output_dir / "table_subgroups.csv", subgroup_rows)
    _write_csv(args.output_dir / "table_quality_efficiency_reductions.csv", reductions)
    _write_csv(args.output_dir / "table_failures.csv", failure_rows)
    _write_csv(args.output_dir / "manual_failure_review_sample.csv", failure_sample)
    _write_csv(args.output_dir / "table_ablations.csv", ablation_rows)
    print(json.dumps({"output": str(args.output_dir), "status": payload["status"], "records": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
