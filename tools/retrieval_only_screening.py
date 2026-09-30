"""Retrieval-only ablation screening, granularity controls and routing diagnostics.

Makes no Groq calls and touches no evaluation split.  Every condition registered
in ``sar_rag.ablations`` that changes only retrieval can be screened here for
free; only conditions whose *answer* behaviour differs need paid evaluation.

The sample is a fresh deterministic stratified draw from the official TRAINING
splits using a seed distinct from the calibration seed, so screening does not
re-use the examples that selected the frozen thresholds.

Usage:
    python tools/retrieval_only_screening.py --per-stratum 5 --seed 20260804
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import replace
from itertools import chain
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sar_rag.ablations import apply_condition  # noqa: E402
from sar_rag.adaptive_pipeline import SarRagSystem  # noqa: E402
from sar_rag.benchmark import _documents  # noqa: E402
from sar_rag.calibration import (  # noqa: E402
    RetrievalOnlyAnswerer, _covered_spans, deterministic_stratified_sample,
)
from sar_rag.config import SarRagConfig  # noqa: E402
from sar_rag.corpus import build_corpus  # noqa: E402
from sar_rag.dataset_adapters import iter_musique_full, iter_qasper  # noqa: E402
from sar_rag.encoding import SentenceTransformerEncoder  # noqa: E402


# Conditions whose effect is entirely inside retrieval.  legacy_multi_call is
# excluded: it changes the online LLM policy and cannot be screened offline.
RETRIEVAL_ONLY_CONDITIONS = (
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
)

BASELINE_GRANULARITY = ("sliding_window", "sliding_window_matched")
BASELINE_KS = (3, 5, 10, 20)


def frozen_proposed() -> dict:
    manifest = json.loads(
        (ROOT / "output" / "calibration" / "frozen_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "training_only_calibration_frozen"
    return manifest["frozen_configuration"]["proposed"]


def _mean(values):
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return sum(clean) / len(clean) if clean else float("nan")


def proposed_row(result, row: dict) -> dict:
    gold = tuple(row["gold_evidence_spans"])
    completed = next(event for event in reversed(result.trace)
                     if event.get("event") == "run_completed")
    values = {
        "final_k": float(result.final_k),
        "context_tokens": float(completed.get("final_context_tokens", 0)),
        "candidates_examined": float(completed.get("candidates_examined", 0)),
        "retrieval_rounds": float(completed.get("retrieval_rounds", 0)),
        "selected_representation_count": float(len(result.selected_representations)),
        "two_representations": float(len(result.selected_representations) == 2),
        "stop_reason": result.stop_reason,
        "selected": "+".join(sorted(result.selected_representations)),
    }
    if not gold:
        values.update({"evidence_recall": float("nan"), "evidence_precision": float("nan"),
                       "complete_evidence": float("nan")})
        return values
    covered, relevant = _covered_spans(result.final_evidence, gold)
    values.update({
        "evidence_recall": len(covered) / len(gold),
        "evidence_precision": sum(relevant) / len(relevant) if relevant else 0.0,
        "complete_evidence": float(len(covered) == len(gold)),
    })
    return values


def baseline_row(artifacts, encoder, row: dict, representation: str, k: int,
                 token_budget: int) -> dict:
    candidates = artifacts.indexes[representation].search(row["query"], 1, k, "adaptive")
    evidence, used = [], 0
    for candidate in candidates:
        chunk = artifacts.chunks_by_id[candidate.chunk_id]
        tokens = encoder.token_count(chunk.evidence_text)
        if used + tokens <= token_budget:
            evidence.append(type("Item", (), {"chunk": chunk})())
            used += tokens
    gold = tuple(row["gold_evidence_spans"])
    values = {
        "final_k": float(len(evidence)), "context_tokens": float(used),
        "candidates_examined": float(len(candidates)), "retrieval_rounds": 1.0,
        "selected_representation_count": 1.0, "two_representations": 0.0,
        "stop_reason": "static_top_k", "selected": representation,
    }
    if not gold:
        values.update({"evidence_recall": float("nan"), "evidence_precision": float("nan"),
                       "complete_evidence": float("nan")})
        return values
    covered, relevant = _covered_spans(evidence, gold)
    values.update({
        "evidence_recall": len(covered) / len(gold),
        "evidence_precision": sum(relevant) / len(relevant) if relevant else 0.0,
        "complete_evidence": float(len(covered) == len(gold)),
    })
    return values


def probe_gold_coverage(artifacts, row: dict, probe_depth: int) -> dict[str, set[int]]:
    """Which gold spans each representation surfaces in its own top-probe."""
    gold = tuple(row["gold_evidence_spans"])
    coverage = {}
    for name, index in artifacts.indexes.items():
        found: set[int] = set()
        if gold:
            for candidate in index.search(row["query"], 1, probe_depth, "probe"):
                chunk = artifacts.chunks_by_id[candidate.chunk_id]
                covered, _ = _covered_spans([type("I", (), {"chunk": chunk})()], gold)
                found |= covered
        coverage[name] = found
    return coverage


def summarize(rows: list[dict]) -> dict:
    numeric = ["evidence_recall", "complete_evidence", "evidence_precision",
               "context_tokens", "final_k", "candidates_examined", "retrieval_rounds",
               "selected_representation_count", "two_representations"]
    output = {name: _mean(row.get(name) for row in rows) for name in numeric}
    output["n"] = len(rows)
    output["n_scored"] = sum(
        1 for row in rows
        if row.get("evidence_recall") is not None
        and not math.isnan(float(row.get("evidence_recall", float("nan"))))
    )
    output["stop_reasons"] = dict(Counter(row["stop_reason"] for row in rows).most_common())
    output["selected_states"] = dict(Counter(row["selected"] for row in rows).most_common(6))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-stratum", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260804,
                        help="distinct from the calibration seed so screening uses fresh training examples")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "output" / "independent_verification")
    args = parser.parse_args()

    qasper = ROOT / "datasets" / "raw" / "qasper" / "data" / "qasper-train-v0.3.json"
    musique = ROOT / "datasets" / "raw" / "musique" / "data" / "musique_full_v1.0_train.jsonl"
    sample, population = deterministic_stratified_sample(
        chain(iter_qasper(qasper, "train"), iter_musique_full(musique, "train")),
        args.per_stratum, args.seed,
    )
    print(json.dumps({"stage": "sampled", "n": len(sample), "strata": len(population)}), flush=True)

    base = SarRagConfig(**frozen_proposed())
    encoder = SentenceTransformerEncoder(str(ROOT / "models" / "all-MiniLM-L6-v2"))
    answerer = RetrievalOnlyAnswerer()

    condition_configs = {}
    for condition in RETRIEVAL_ONLY_CONDITIONS:
        config, _, engine = apply_condition(base, 20, condition)
        assert engine == "one_call"
        condition_configs[condition] = config
    proposed_rows: dict[str, list[dict]] = {name: [] for name in RETRIEVAL_ONLY_CONDITIONS}
    baseline_rows: dict[tuple[str, int], list[dict]] = {
        (rep, k): [] for rep in BASELINE_GRANULARITY for k in BASELINE_KS
    }
    routing: list[dict] = []
    granularity_stats: dict[str, list[float]] = {name: [] for name in BASELINE_GRANULARITY}

    all_views = tuple(sorted(set(
        view for config in condition_configs.values()
        for view in config.enabled_representations
    )))

    def restrict(artifacts, views: tuple[str, ...]):
        """A view over already-built indexes; avoids re-embedding for subset conditions."""
        if set(views) == set(artifacts.indexes):
            return artifacts
        indexes = {name: artifacts.indexes[name] for name in views}
        chunks = {chunk.chunk_id: chunk
                  for index in indexes.values() for chunk in index.chunks}
        return type(artifacts)(artifacts.units_by_id, chunks, indexes)

    for position, row in enumerate(sample, 1):
        documents = _documents(row)
        # Two builds per example: the four adaptive views, and the two baseline
        # granularities.  Every condition is then evaluated against a restricted
        # view of the same indexes.
        proposed_artifacts = build_corpus(documents, encoder, all_views)
        baseline_artifacts = build_corpus(documents, encoder, BASELINE_GRANULARITY)
        for name in BASELINE_GRANULARITY:
            chunks = baseline_artifacts.indexes[name].chunks
            granularity_stats[name].extend(float(chunk.word_count) for chunk in chunks)

        results = {}
        for condition, config in condition_configs.items():
            subset = restrict(proposed_artifacts, config.enabled_representations)
            result = SarRagSystem(subset, encoder, answerer, config).ask(row["query"])
            results[condition] = result
            proposed_rows[condition].append(proposed_row(result, row))

        for rep in BASELINE_GRANULARITY:
            for k in BASELINE_KS:
                baseline_rows[(rep, k)].append(baseline_row(
                    baseline_artifacts, encoder, row, rep, k, base.max_final_context_tokens
                ))

        # Routing diagnostics reuse the primary condition's result and indexes.
        full_result = results["full"]
        coverage = probe_gold_coverage(
            restrict(proposed_artifacts, base.enabled_representations),
            row, base.probe_depth,
        )
        selected = set(full_result.selected_representations)
        selected_cov: set[int] = set().union(*[coverage[n] for n in selected]) if selected else set()
        unselected = set(base.enabled_representations) - selected
        unselected_cov: set[int] = set().union(*[coverage[n] for n in unselected]) if unselected else set()
        first_only = sorted(selected)[:1]
        first_cov: set[int] = set().union(*[coverage[n] for n in first_only]) if first_only else set()
        gold_n = len(row["gold_evidence_spans"])
        routing.append({
            "example_id": row["example_id"], "dataset": row["dataset"],
            "gold_spans": gold_n,
            "selected": "+".join(sorted(selected)),
            "selected_count": len(selected),
            "probe_gold_selected": len(selected_cov),
            "probe_gold_unselected_unique": len(unselected_cov - selected_cov),
            "unselected_has_unique_gold": float(bool(unselected_cov - selected_cov)),
            "second_view_unique_gold": len(selected_cov - first_cov) if len(selected) == 2 else 0,
            "second_view_added_gold": float(bool(selected_cov - first_cov)) if len(selected) == 2 else 0.0,
        })
        print(json.dumps({"stage": "screened", "completed": position, "total": len(sample)}),
              flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_out = []
    for condition in RETRIEVAL_ONLY_CONDITIONS:
        summary = summarize(proposed_rows[condition])
        rows_out.append({"arm": "sar_rag", "condition": condition,
                         "granularity_words": None, "static_k": None, **summary})
    for rep in BASELINE_GRANULARITY:
        for k in BASELINE_KS:
            summary = summarize(baseline_rows[(rep, k)])
            rows_out.append({
                "arm": "traditional", "condition": f"{rep}_k{k}",
                "granularity_words": round(_mean(granularity_stats[rep]), 1),
                "static_k": k, **summary,
            })

    fields = ["arm", "condition", "granularity_words", "static_k", "n", "n_scored",
              "evidence_recall", "complete_evidence", "evidence_precision",
              "context_tokens", "final_k", "candidates_examined", "retrieval_rounds",
              "selected_representation_count", "two_representations",
              "stop_reasons", "selected_states"]
    with (args.output_dir / "RETRIEVAL_ONLY_ABLATIONS.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows_out:
            writer.writerow({**row,
                             "stop_reasons": json.dumps(row.get("stop_reasons", {})),
                             "selected_states": json.dumps(row.get("selected_states", {}))})

    two_view = [r for r in routing if r["selected_count"] == 2]
    scored = [r for r in routing if r["gold_spans"] > 0]
    diagnostics = {
        "scope": "training-only retrieval screening; no Groq calls; no evaluation split touched",
        "sample": {"n": len(sample), "seed": args.seed, "per_stratum": args.per_stratum,
                   "strata": len(population),
                   "note": "seed differs from the calibration seed so screening does not reuse threshold-selection examples"},
        "granularity_control": {
            name: {
                "mean_chunk_words": round(_mean(values), 2),
                "median_chunk_words": round(statistics.median(values), 2) if values else None,
                "chunks": len(values),
            } for name, values in granularity_stats.items()
        },
        "representation_selection": {
            "one_view_percent": 100 * sum(r["selected_count"] == 1 for r in routing) / len(routing),
            "two_view_percent": 100 * len(two_view) / len(routing),
            "state_counts": dict(Counter(r["selected"] for r in routing).most_common()),
            "first_view_counts": dict(Counter(
                r["selected"].split("+")[0] for r in routing).most_common()),
            "second_view_added_unique_gold_percent_of_two_view": (
                100 * _mean(r["second_view_added_gold"] for r in two_view) if two_view else None
            ),
            "unselected_view_holds_unique_gold_percent": (
                100 * _mean(r["unselected_has_unique_gold"] for r in scored) if scored else None
            ),
            "n_two_view": len(two_view), "n_scored": len(scored),
        },
        "conditions_screened": list(RETRIEVAL_ONLY_CONDITIONS),
        "conditions_requiring_paid_evaluation": ["legacy_multi_call"],
    }
    (args.output_dir / "REPRESENTATION_SELECTION_DIAGNOSTICS.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "REPRESENTATION_ROUTING_PER_EXAMPLE.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(routing[0]))
        writer.writeheader(); writer.writerows(routing)
    print(json.dumps({"output": str(args.output_dir), "conditions": len(rows_out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
