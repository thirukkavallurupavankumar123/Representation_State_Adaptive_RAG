from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
from hashlib import sha256
import heapq
from itertools import chain
import json
import math
from pathlib import Path
from statistics import mean
from typing import Iterable

import numpy as np

from .benchmark import _documents
from .config import SarRagConfig
from .corpus import build_corpus
from .dataset_adapters import iter_musique_full, iter_qasper
from .encoding import SentenceTransformerEncoder
from .evaluation import normalize_answer
from .models import FinalAnswerDecision
from .pipeline import SarRagSystem


class RetrievalOnlyAnswerer:
    """Calibration never calls an LLM and never uses answer labels."""

    def generate(self, query, evidence):
        return FinalAnswerDecision("retrieval-only calibration", "INSUFFICIENT", ())


def _hash_key(seed: int, example_id: str) -> int:
    return int(sha256(f"{seed}|{example_id}".encode()).hexdigest(), 16)


def _hop_bucket(row: dict) -> str:
    source = str(row.get("metadata", {}).get("source_id", ""))
    for value in ("2hop", "3hop", "4hop"):
        if value in source:
            return value
    return "unknown_hops"


def _length_bucket(row: dict) -> str:
    words = sum(len(document["text"].split()) for document in row["documents"])
    if words < 2_000:
        return "short"
    if words < 6_000:
        return "medium"
    return "long"


def stratum(row: dict) -> str:
    answerability = row.get("gold_answerability", "UNKNOWN")
    dataset = row["dataset"]
    if dataset == "musique_full":
        return f"{dataset}|{answerability}|{_hop_bucket(row)}"
    return f"{dataset}|{answerability}|{_length_bucket(row)}"


def deterministic_stratified_sample(rows: Iterable[dict], per_stratum: int,
                                    seed: int) -> tuple[list[dict], dict[str, int]]:
    """Streaming bottom-hash sample: independent of source iteration ordering."""
    heaps: dict[str, list[tuple[int, str, dict]]] = defaultdict(list)
    population: dict[str, int] = defaultdict(int)
    for row in rows:
        group = stratum(row)
        population[group] += 1
        key = _hash_key(seed, row["example_id"])
        entry = (-key, row["example_id"], row)
        heap = heaps[group]
        if len(heap) < per_stratum:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)
    selected = [entry[2] for heap in heaps.values() for entry in heap]
    selected.sort(key=lambda row: (stratum(row), _hash_key(seed, row["example_id"])))
    return selected, dict(sorted(population.items()))


def diverse_smoke_sample(rows: list[dict], limit: int) -> list[dict]:
    """Round-robin across dataset and answerability for implementation smoke runs."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row.get("gold_answerability", "UNKNOWN"))].append(row)
    output = []
    ordered_keys = sorted(groups)
    while len(output) < limit and any(groups.values()):
        for key in ordered_keys:
            if groups[key] and len(output) < limit:
                output.append(groups[key].pop(0))
    return output


def partition_training_sample(rows: list[dict], seed: int,
                              selection_fraction: float) -> dict[str, str]:
    """Deterministic within-stratum tuning/selection partition."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[stratum(row)].append(row)
    partitions = {}
    for group, values in grouped.items():
        values.sort(key=lambda row: _hash_key(seed + 1, row["example_id"]))
        if len(values) == 1:
            selection_count = 1
        else:
            selection_count = max(1, round(len(values) * selection_fraction))
            selection_count = min(selection_count, len(values) - 1)
        selection_ids = {row["example_id"] for row in values[:selection_count]}
        for row in values:
            partitions[row["example_id"]] = (
                "selection" if row["example_id"] in selection_ids else "tuning"
            )
    return partitions


def _covered_spans(evidence, gold_spans: tuple[str, ...]) -> tuple[set[int], list[int]]:
    covered: set[int] = set()
    relevant = []
    normalized_gold = [normalize_answer(span) for span in gold_spans]
    for item in evidence:
        text = normalize_answer(item.chunk.evidence_text)
        matches = {
            index for index, gold in enumerate(normalized_gold)
            if gold and (gold in text or (len(text.split()) >= 5 and text in gold))
        }
        covered.update(matches)
        relevant.append(int(bool(matches)))
    return covered, relevant


def retrieval_row(result, row: dict) -> dict[str, float]:
    gold = tuple(row["gold_evidence_spans"])
    completed = next(event for event in reversed(result.trace)
                     if event.get("event") == "run_completed")
    values = {
        "answerable": float(row.get("gold_answerability") == "ANSWERABLE"),
        "final_k": float(result.final_k),
        "context_tokens": float(completed.get("final_context_tokens", 0)),
        "candidates_examined": float(completed.get("candidates_examined", 0)),
        "selected_representations": float(len(result.selected_representations)),
    }
    if not gold:
        values.update({
            "evidence_recall": float("nan"),
            "evidence_precision": float("nan"),
            "complete_evidence": float("nan"),
        })
        return values
    covered, relevant = _covered_spans(result.final_evidence, gold)
    values.update({
        "evidence_recall": len(covered) / len(gold),
        "evidence_precision": sum(relevant) / len(relevant) if relevant else 0.0,
        "complete_evidence": float(len(covered) == len(gold)),
    })
    return values


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    output = {}
    for key in rows[0] if rows else ():
        clean = [float(row[key]) for row in rows if not math.isnan(float(row[key]))]
        output[key] = mean(clean) if clean else float("nan")
    answerable = [row for row in rows if row["answerable"] == 1]
    insufficient = [row for row in rows if row["answerable"] == 0]
    output["answerable_context_tokens"] = mean([row["context_tokens"] for row in answerable]) if answerable else float("nan")
    output["insufficient_context_tokens"] = mean([row["context_tokens"] for row in insufficient]) if insufficient else float("nan")
    output["insufficient_candidates_examined"] = mean([row["candidates_examined"] for row in insufficient]) if insufficient else float("nan")
    return output


def choose_configuration(results: list[dict], recall_tolerance: float = 0.01) -> dict:
    """Constrained lexicographic choice; avoids arbitrary weighted metric sums."""
    best_recall = max(item["metrics"]["evidence_recall"] for item in results)
    eligible = [item for item in results
                if item["metrics"]["evidence_recall"] >= best_recall - recall_tolerance]
    eligible.sort(key=lambda item: (
        -item["metrics"]["complete_evidence"],
        -item["metrics"]["evidence_precision"],
        item["metrics"]["context_tokens"],
        item["metrics"]["candidates_examined"],
        json.dumps(item["parameters"], sort_keys=True),
    ))
    return eligible[0]


def tuning_shortlist(results: list[dict], recall_tolerance: float = 0.01) -> list[dict]:
    best_recall = max(item["tuning_metrics"]["evidence_recall"] for item in results)
    return [item for item in results
            if item["tuning_metrics"]["evidence_recall"] >= best_recall - recall_tolerance]


def select_from_heldback_training(results: list[dict]) -> dict:
    candidates = [
        {"parameters": item["parameters"], "metrics": item["selection_metrics"],
         "tuning_metrics": item["tuning_metrics"]}
        for item in tuning_shortlist(results)
    ]
    selected = choose_configuration(candidates)
    original = next(item for item in results if item["parameters"] == selected["parameters"])
    return original


def parameter_candidates(trials: int, seed: int) -> list[dict]:
    anchors = [{
        "rrf_k0": 3,
        "representation_marginal_gain_threshold": 0.10,
        "similarity_threshold": 0.70,
        "similarity_gap_threshold": 0.75,
        "novelty_threshold": 0.10,
        "redundancy_threshold": 0.92,
        "marginal_utility_threshold": 0.05,
        "empty_batch_patience": 2,
    }]
    rng = np.random.default_rng(seed)
    values = list(anchors)
    for _ in range(max(0, trials - len(anchors))):
        values.append({
            "rrf_k0": int(rng.choice([1, 3, 10, 60])),
            "representation_marginal_gain_threshold": round(float(rng.uniform(0.03, 0.25)), 4),
            "similarity_threshold": round(float(rng.uniform(0.50, 0.95)), 4),
            "similarity_gap_threshold": round(float(rng.uniform(0.25, 2.50)), 4),
            "novelty_threshold": round(float(rng.uniform(0.03, 0.30)), 4),
            "redundancy_threshold": round(float(rng.uniform(0.85, 0.98)), 4),
            "marginal_utility_threshold": round(float(rng.uniform(0.02, 0.20)), 4),
            "empty_batch_patience": int(rng.choice([1, 2, 3])),
        })
    unique = {json.dumps(value, sort_keys=True): value for value in values}
    return [unique[key] for key in sorted(unique)]


def _baseline_metrics(artifacts, encoder, row: dict, k: int, token_budget: int) -> dict:
    candidates = artifacts.indexes["sliding_window"].search(row["query"], 1, k, "adaptive")
    evidence, used = [], 0
    for candidate in candidates:
        chunk = artifacts.chunks_by_id[candidate.chunk_id]
        tokens = encoder.token_count(chunk.evidence_text)
        if used + tokens <= token_budget:
            evidence.append(type("Item", (), {"chunk": chunk})())
            used += tokens
    gold = tuple(row["gold_evidence_spans"])
    if gold:
        covered, relevant = _covered_spans(evidence, gold)
        recall = len(covered) / len(gold)
        precision = sum(relevant) / len(relevant) if relevant else 0.0
        complete = float(len(covered) == len(gold))
    else:
        recall = precision = complete = float("nan")
    return {
        "answerable": float(row.get("gold_answerability") == "ANSWERABLE"),
        "evidence_recall": recall,
        "evidence_precision": precision,
        "complete_evidence": complete,
        "final_k": float(len(evidence)),
        "context_tokens": float(used),
        "candidates_examined": float(len(candidates)),
        "selected_representations": 1.0,
    }


def _checksum(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Training-only SAR-RAG calibration")
    parser.add_argument("--qasper-train", type=Path, required=True)
    parser.add_argument("--musique-train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-stratum", type=int, default=5)
    parser.add_argument("--search-trials", type=int, default=32)
    parser.add_argument("--selection-fraction", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--baseline-k", type=int, nargs="+", default=[3, 5, 8, 10, 15, 20])
    parser.add_argument("--smoke-limit", type=int,
                        help="truncate the selected sample for code verification; output cannot freeze parameters")
    args = parser.parse_args(argv)
    if args.per_stratum < 1 or args.search_trials < 1:
        raise SystemExit("sample and search sizes must be positive")
    if not 0 < args.selection_fraction < 0.5:
        raise SystemExit("selection fraction must be between 0 and 0.5")

    source_rows = chain(
        iter_qasper(args.qasper_train, "train"),
        iter_musique_full(args.musique_train, "train"),
    )
    sample, population = deterministic_stratified_sample(source_rows, args.per_stratum, args.seed)
    if args.smoke_limit is not None:
        if args.smoke_limit < 1:
            raise SystemExit("smoke limit must be positive")
        sample = diverse_smoke_sample(sample, args.smoke_limit)
    partitions = partition_training_sample(sample, args.seed, args.selection_fraction)

    base = SarRagConfig()
    local = Path(base.local_embedding_path or "")
    encoder = SentenceTransformerEncoder(str(local) if local.is_dir() else base.embedding_model,
                                         revision=base.embedding_revision)
    answerer = RetrievalOnlyAnswerer()
    corpora = []
    corpus_cache = {}
    for position, row in enumerate(sample, 1):
        documents = _documents(row)
        corpus_key = sha256(json.dumps(
            [{"document_id": value.document_id, "version_id": value.version_id}
             for value in documents], sort_keys=True
        ).encode()).hexdigest()
        if corpus_key not in corpus_cache:
            corpus_cache[corpus_key] = (
                build_corpus(documents, encoder, base.enabled_representations),
                build_corpus(documents, encoder, ("sliding_window",)),
            )
        proposed, baseline = corpus_cache[corpus_key]
        corpora.append((row, proposed, baseline))
        print(json.dumps({"stage": "indexing", "completed": position, "total": len(sample),
                          "unique_corpora": len(corpus_cache)}), flush=True)

    proposed_results = []
    for parameters in parameter_candidates(args.search_trials, args.seed):
        config = replace(base, **parameters)
        rows = []
        for row, artifacts, _ in corpora:
            result = SarRagSystem(artifacts, encoder, answerer, config).ask(row["query"])
            rows.append(retrieval_row(result, row))
        tuning = [value for value, (row, _, _) in zip(rows, corpora)
                  if partitions[row["example_id"]] == "tuning"]
        selection = [value for value, (row, _, _) in zip(rows, corpora)
                     if partitions[row["example_id"]] == "selection"]
        # A tiny smoke sample may contain only selection members. It remains non-freezing.
        tuning = tuning or rows
        selection = selection or rows
        proposed_results.append({
            "parameters": parameters,
            "tuning_metrics": summarize(tuning),
            "selection_metrics": summarize(selection),
        })
        print(json.dumps({"stage": "proposed_search", "completed": len(proposed_results),
                          "total": args.search_trials}), flush=True)
    proposed_choice = select_from_heldback_training(proposed_results)

    baseline_results = []
    for k in sorted(set(args.baseline_k)):
        rows = [_baseline_metrics(baseline, encoder, row, k, base.max_final_context_tokens)
                for row, _, baseline in corpora]
        tuning = [value for value, (row, _, _) in zip(rows, corpora)
                  if partitions[row["example_id"]] == "tuning"] or rows
        selection = [value for value, (row, _, _) in zip(rows, corpora)
                     if partitions[row["example_id"]] == "selection"] or rows
        baseline_results.append({
            "parameters": {"static_k": k},
            "tuning_metrics": summarize(tuning),
            "selection_metrics": summarize(selection),
        })
    baseline_choice = select_from_heldback_training(baseline_results)

    is_smoke = args.smoke_limit is not None
    manifest = {
        "status": "smoke_not_frozen" if is_smoke else "training_only_calibration_frozen",
        "seed": args.seed,
        "selection_rule": {
            "description": "shortlist configurations within 0.01 of best tuning evidence recall; choose on held-back training selection partition by complete-evidence rate, precision, context tokens, then candidates",
            "recall_tolerance": 0.01,
            "selection_fraction": args.selection_fraction,
            "answer_generation_used": False,
        },
        "dataset_sources": {
            "qasper_train": {"path": str(args.qasper_train), "sha256": _checksum(args.qasper_train)},
            "musique_train": {"path": str(args.musique_train), "sha256": _checksum(args.musique_train)},
        },
        "population_strata": population,
        "sample_size": len(sample),
        "sample_ids": [row["example_id"] for row in sample],
        "sample_partitions": partitions,
        "embedding": {"model": base.embedding_model, "revision": base.embedding_revision},
        "proposed_search_space": {
            "method": "seeded random search plus registered anchor",
            "trials": len(proposed_results),
            "bounds": {
                "rrf_k0": [1, 3, 10, 60],
                "representation_marginal_gain_threshold": [0.03, 0.25],
                "similarity_threshold": [0.50, 0.95],
                "similarity_gap_threshold": [0.25, 2.50],
                "novelty_threshold": [0.03, 0.30],
                "redundancy_threshold": [0.85, 0.98],
                "marginal_utility_threshold": [0.02, 0.20],
                "empty_batch_patience": [1, 2, 3],
            },
        },
        "selected_proposed_parameters": proposed_choice["parameters"],
        "selected_baseline_parameters": baseline_choice["parameters"],
        "frozen_configuration": None if is_smoke else {
            "proposed": proposed_choice["parameters"],
            "baseline": baseline_choice["parameters"],
        },
        "chosen_proposed_tuning_metrics": proposed_choice["tuning_metrics"],
        "chosen_proposed_selection_metrics": proposed_choice["selection_metrics"],
        "chosen_baseline_tuning_metrics": baseline_choice["tuning_metrics"],
        "chosen_baseline_selection_metrics": baseline_choice["selection_metrics"],
        "all_proposed_trials": proposed_results,
        "all_baseline_trials": baseline_results,
        "limitations": [
            "Calibration optimizes retrieval only; it makes no API calls and does not use gold answer text.",
            "Gold-span containment is imperfect for paraphrased evidence.",
            "The confirmatory study must not alter this manifest after validation/test inspection.",
            *( ["Smoke-limit output is implementation verification and must not be used to freeze parameters."]
               if args.smoke_limit is not None else [] ),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2,
                                      allow_nan=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sample_size": len(sample),
                      "proposed_trials": len(proposed_results),
                      "baseline_trials": len(baseline_results)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
