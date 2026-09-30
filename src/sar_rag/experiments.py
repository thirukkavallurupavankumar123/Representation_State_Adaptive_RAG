from __future__ import annotations

import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
import math
import os
import re
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from .benchmark import _documents, _load_jsonl
from .ablations import CONDITIONS, apply_condition
from .config import SarRagConfig
from .encoding import SentenceTransformerEncoder
from .environment import load_project_env
from .evaluation import exact_match, normalize_answer, token_f1, token_prf
from .groq import (
    GroqAnswerGenerator, GroqClient, GroqEvidenceJudge, GroqOneCallAnswerGenerator,
    GroqReformulator, GroqRequirementExtractor,
)
from .pipeline import LlmJudgedSarRagSystem, SarRagSystem, build_corpus
from .traditional import TraditionalRagConfig, TraditionalRagSystem


PAIR_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "A": {"type": "object", "properties": {
            "faithfulness": {"type": "number"}, "answer_relevancy": {"type": "number"},
        }, "required": ["faithfulness", "answer_relevancy"], "additionalProperties": False},
        "B": {"type": "object", "properties": {
            "faithfulness": {"type": "number"}, "answer_relevancy": {"type": "number"},
        }, "required": ["faithfulness", "answer_relevancy"], "additionalProperties": False},
    },
    "required": ["A", "B"], "additionalProperties": False,
}


def _usage(log: list[dict], cost_rates: tuple[float, float, float] | None) -> dict[str, float]:
    prompt = sum(x["prompt_tokens"] for x in log)
    completion = sum(x["completion_tokens"] for x in log)
    cached = sum(x["cached_tokens"] for x in log)
    uncached = max(prompt - cached, 0)
    estimated_cost = float("nan")
    if cost_rates is not None:
        input_rate, cached_rate, output_rate = cost_rates
        estimated_cost = (
            uncached * input_rate * 1e-6
            + cached * cached_rate * 1e-6
            + completion * output_rate * 1e-6
        )
    return {
        "api_calls": float(len(log)),
        "prompt_tokens": float(prompt),
        "completion_tokens": float(completion),
        "total_tokens": float(sum(x["total_tokens"] for x in log)),
        "cached_tokens": float(cached),
        "api_latency_seconds": sum(x["latency_seconds"] for x in log),
        "estimated_cost_usd": estimated_cost,
    }


def _attempt_usage(log: list[dict]) -> dict[str, float]:
    return {
        "api_attempts": float(len(log)),
        "failed_api_attempts": float(sum(item.get("status") != "success" for item in log)),
    }


def _windows_peak_working_set_bytes() -> int | None:
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        # HANDLE is pointer-sized.  Leaving ctypes' default 32-bit int restype on
        # GetCurrentProcess truncated the (HANDLE)-1 pseudo-handle, so on 64-bit
        # Windows GetProcessMemoryInfo always failed with ERROR_INVALID_HANDLE and
        # the peak working set was silently recorded as null.
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = Counters(); counters.cb = ctypes.sizeof(Counters)
        if not psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            return None
        return int(counters.PeakWorkingSetSize)
    except (AttributeError, OSError):
        return None


def _relevance(item, gold_spans: tuple[str, ...]) -> tuple[bool, set[int]]:
    text = normalize_answer(item.chunk.evidence_text)
    covered = set()
    for index, span in enumerate(gold_spans):
        gold = normalize_answer(span)
        if gold and (gold in text or (len(text.split()) >= 5 and text in gold)):
            covered.add(index)
    return bool(covered), covered


def retrieval_metrics(evidence, gold_spans: tuple[str, ...], cutoffs=(1, 3, 5, 10, 20)) -> dict[str, float]:
    if not gold_spans:
        # Retrieval effectiveness has no denominator when the benchmark declares
        # that no relevant evidence exists.  Treating abstention as recall=1 or
        # nDCG=1 would inflate aggregate retrieval quality on unanswerable items.
        undefined = {}
        for cutoff in cutoffs:
            undefined[f"retrieval_precision_at_{cutoff}"] = float("nan")
            undefined[f"retrieval_recall_at_{cutoff}"] = float("nan")
            undefined[f"retrieval_ndcg_at_{cutoff}"] = float("nan")
        undefined.update({
            "retrieval_mrr": float("nan"),
            "evidence_set_precision": float("nan"),
            "evidence_set_recall": float("nan"),
            "complete_evidence_recovery": float("nan"),
        })
        return undefined
    judgments = [_relevance(item, gold_spans) for item in evidence]
    binary = [int(value[0]) for value in judgments]
    metrics = {}
    for cutoff in cutoffs:
        prefix = judgments[:cutoff]
        covered = set().union(*(value[1] for value in prefix)) if prefix else set()
        metrics[f"retrieval_precision_at_{cutoff}"] = sum(value[0] for value in prefix) / cutoff
        metrics[f"retrieval_recall_at_{cutoff}"] = len(covered) / len(gold_spans)
        dcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(binary[:cutoff]))
        ideal_relevant = min(len(gold_spans), cutoff)
        idcg = sum(1 / math.log2(rank + 2) for rank in range(ideal_relevant))
        metrics[f"retrieval_ndcg_at_{cutoff}"] = dcg / idcg
    first = next((index for index, value in enumerate(binary, 1) if value), None)
    metrics["retrieval_mrr"] = 1 / first if first else 0.0
    metrics["evidence_set_precision"] = sum(binary) / len(binary) if binary else 0.0
    all_covered = set().union(*(value[1] for value in judgments)) if judgments else set()
    metrics["evidence_set_recall"] = len(all_covered) / len(gold_spans)
    metrics["complete_evidence_recovery"] = float(len(all_covered) == len(gold_spans))
    return metrics


def method_metrics(result, row: dict, wall_seconds: float, usage: dict) -> dict[str, float]:
    gold_answers = tuple(row["gold_answers"])
    gold_spans = tuple(row["gold_evidence_spans"])
    values = retrieval_metrics(result.final_evidence, gold_spans)
    answer_precision, answer_recall, answer_f1 = (
        token_prf(result.answer, gold_answers) if gold_answers else (float("nan"),) * 3
    )
    completed = next((event for event in reversed(result.trace)
                      if event.get("event") == "run_completed"), {})
    values.update({
        "answer_exact_match": exact_match(result.answer, gold_answers) if gold_answers else float("nan"),
        "answer_token_precision": answer_precision,
        "answer_token_recall": answer_recall,
        "answer_token_f1": answer_f1,
        "answerability_correct": float(
            result.answerability == row["gold_answerability"]
        ) if row.get("gold_answerability") else float("nan"),
        "predicted_answerable": float(result.answerability != "INSUFFICIENT"),
        "gold_answerable": float(row.get("gold_answerability") == "ANSWERABLE"),
        "latency_seconds": wall_seconds,
        "final_k": float(result.final_k),
        "context_words": float(sum(item.chunk.word_count for item in result.final_evidence)),
        "context_tokens": float(completed.get("final_context_tokens",
                                               completed.get("context_tokens", 0))),
        "candidates_examined": float(completed.get("candidates_examined", result.final_k)),
        "retrieved_before_deduplication": float(
            completed.get("retrieved_before_deduplication", result.final_k)
        ),
        "retrieval_rounds": float(completed.get("retrieval_rounds", 0)),
    })
    values.update(usage)
    return values


def paired_quality_judge(client: GroqClient, query: str, results: dict, reverse: bool) -> dict:
    names = ["traditional", "proposed"]
    if reverse:
        names.reverse()
    labels = dict(zip(("A", "B"), names))
    blocks = []
    for label, name in labels.items():
        result = results[name]
        evidence = "\n".join(f"[E{i}] {x.chunk.evidence_text}" for i, x in enumerate(result.final_evidence, 1))
        blocks.append(f"SYSTEM {label}\nANSWER:\n{result.answer}\nEVIDENCE:\n{evidence}")
    prompt = f"""Independently score both systems from 0 to 1. Faithfulness is the
fraction of answer claims supported by that system's supplied evidence. Answer relevancy
is how directly the answer addresses the query. Do not use outside knowledge and do not
prefer longer answers. Return calibrated decimal scores.

QUERY: {query}
{chr(10).join(blocks)}"""
    data = json.loads(client.complete(
        [{"role": "system", "content": "You are a blinded paired RAG evaluator."},
         {"role": "user", "content": prompt}],
        "paired_rag_quality", PAIR_JUDGE_SCHEMA, 600,
    ))
    return {name: data[label] for label, name in labels.items()}


def _mean(values):
    clean = [float(x) for x in values if not math.isnan(float(x))]
    return sum(clean) / len(clean) if clean else float("nan")


def paired_statistics(rows: list[dict], resamples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    metrics = sorted(set(rows[0]["traditional"]) & set(rows[0]["proposed"])) if rows else []
    output = {}
    for metric in metrics:
        pairs = [(x["traditional"][metric], x["proposed"][metric]) for x in rows]
        pairs = [(a, b) for a, b in pairs if not math.isnan(float(a)) and not math.isnan(float(b))]
        if not pairs:
            continue
        differences = np.asarray([b - a for a, b in pairs], dtype=float)
        # Do not allocate resamples x examples for the complete benchmark.  Keep
        # temporary draws bounded while retaining all resampled means.
        batch_size = max(1, min(resamples, 2_000_000 // len(differences)))
        boot = np.empty(resamples, dtype=float)
        randomized = np.empty(resamples, dtype=float)
        for start in range(0, resamples, batch_size):
            stop = min(start + batch_size, resamples)
            size = stop - start
            indices = rng.integers(0, len(differences), size=(size, len(differences)))
            boot[start:stop] = differences[indices].mean(axis=1)
            signs = rng.integers(0, 2, size=(size, len(differences)), dtype=np.int8)
            signs = signs * 2 - 1
            randomized[start:stop] = (differences * signs).mean(axis=1)
        observed = float(differences.mean())
        p_value = float((np.sum(np.abs(randomized) >= abs(observed)) + 1) / (resamples + 1))
        output[metric] = {
            "traditional_mean": _mean(a for a, _ in pairs),
            "proposed_mean": _mean(b for _, b in pairs),
            "paired_difference_proposed_minus_traditional": observed,
            "bootstrap_95_ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "paired_randomization_p_value": p_value,
            "paired_effect_size_dz": (
                observed / float(differences.std(ddof=1))
                if len(differences) > 1 and differences.std(ddof=1) > 0 else 0.0
            ),
            "paired_n": len(pairs),
        }
    return output


def exact_mcnemar_p_value(a: list[int], b: list[int]) -> dict[str, float | int]:
    traditional_only = sum(x == 1 and y == 0 for x, y in zip(a, b))
    proposed_only = sum(x == 0 and y == 1 for x, y in zip(a, b))
    discordant = traditional_only + proposed_only
    if discordant == 0:
        p_value = 1.0
    else:
        smaller = min(traditional_only, proposed_only)
        lower_tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2 ** discordant)
        p_value = min(1.0, 2 * lower_tail)
    return {
        "traditional_only_correct": traditional_only,
        "proposed_only_correct": proposed_only,
        "discordant_pairs": discordant,
        "exact_mcnemar_p_value": p_value,
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted, running = {}, 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[name] = running
    return adjusted


def classification_summary(rows: list[dict], method: str) -> dict[str, float]:
    pairs = [(int(x[method]["gold_answerable"]), int(x[method]["predicted_answerable"])) for x in rows]
    tp = sum(a == 1 and b == 1 for a, b in pairs)
    tn = sum(a == 0 and b == 0 for a, b in pairs)
    fp = sum(a == 0 and b == 1 for a, b in pairs)
    fn = sum(a == 1 and b == 0 for a, b in pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": (tp + tn) / len(pairs) if pairs else 0.0,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "false_answer_rate_on_insufficient": fp / (fp + tn) if fp + tn else float("nan"),
        "correct_abstention_rate": tn / (tn + fp) if tn + fp else float("nan"),
        "incorrect_abstention_rate_on_answerable": fn / (fn + tp) if fn + tp else float("nan"),
    }


def _example_descriptors(row: dict) -> dict:
    document_words = sum(len(document.get("text", "").split()) for document in row["documents"])
    answer_words = max((len(answer.split()) for answer in row.get("gold_answers", [])), default=0)
    source_id = str(row.get("metadata", {}).get("source_id", ""))
    hop_match = re.match(r"(\d+)hop", source_id)
    if document_words < 4_000:
        length_group = "short"
    elif document_words < 10_000:
        length_group = "medium"
    else:
        length_group = "long"
    return {
        "document_words": document_words,
        "document_count": len(row["documents"]),
        "document_length_group": length_group,
        "hop_count": int(hop_match.group(1)) if hop_match else None,
        "gold_answer_words": answer_words,
        "answer_length_group": "none" if answer_words == 0 else ("short" if answer_words <= 5 else "long"),
        "gold_evidence_count": len(row.get("gold_evidence_spans", [])),
    }


def _trace_summary(result) -> dict:
    scored = [event for event in result.trace if event.get("event") == "candidate_scored_without_llm"]
    pauses = [event for event in result.trace if event.get("event") == "representation_paused"]
    return {
        "stop_reason": result.stop_reason,
        "rejection_counts": dict(Counter(
            event["rejection_reason"] for event in scored if not event.get("accepted")
        )),
        "pause_reasons": dict(Counter(event.get("reason") for event in pauses)),
        "representation_profiles": {
            name: {
                "selected": profile.selected,
                "coverage": profile.normalized_coverage_score,
                "marginal_gain": profile.marginal_gain,
                "mean_probe_words": profile.mean_probe_words,
            } for name, profile in result.representation_profiles.items()
        },
    }


def _representation_gold_diagnostic(system, query: str, gold_spans: tuple[str, ...],
                                    probe_depth: int) -> dict:
    output = {}
    for representation, index in system.artifacts.indexes.items():
        probe = index.search(query, 1, probe_depth, "probe")
        covered = set()
        for candidate in probe:
            chunk = system.artifacts.chunks_by_id[candidate.chunk_id]
            covered.update(_relevance(type("ProbeItem", (), {"chunk": chunk})(), gold_spans)[1])
        output[representation] = {
            "gold_span_indices_in_top_probe": sorted(covered),
            "gold_span_count_in_top_probe": len(covered),
        }
    return output


def save_graphs(statistics: dict, output: Path) -> None:
    # Figures are written to disk and never displayed.  Selecting the default
    # interactive backend made an unrelated broken Tk installation abort the run
    # after every paid API call had already been made.
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    quality_names = [x for x in ("answer_token_f1", "evidence_set_recall", "faithfulness",
                                 "answer_relevancy") if x in statistics]
    efficiency_names = [x for x in ("latency_seconds", "total_tokens", "estimated_cost_usd")
                        if x in statistics]
    if not quality_names and not efficiency_names:
        return
    figure, axes = plt.subplots(1, 1 + len(efficiency_names), figsize=(5 * (1 + len(efficiency_names)), 5))
    axes = np.atleast_1d(axes)
    if quality_names:
        x = np.arange(len(quality_names))
        axes[0].bar(x - .18, [statistics[name]["traditional_mean"] for name in quality_names], .36,
                    label="Traditional")
        axes[0].bar(x + .18, [statistics[name]["proposed_mean"] for name in quality_names], .36,
                    label="Proposed")
        axes[0].set_xticks(x, [
            name.replace("_", "\n") + f"\n(n={statistics[name]['paired_n']})"
            for name in quality_names
        ])
        axes[0].set_ylim(0, 1.08)
        axes[0].set_title("Quality (descriptive pilot)")
        axes[0].legend()
    for axis, name in zip(axes[1:], efficiency_names):
        values = [statistics[name]["traditional_mean"], statistics[name]["proposed_mean"]]
        bars = axis.bar(["Traditional", "Proposed"], values, color=["#1f77b4", "#ff7f0e"])
        axis.bar_label(bars, fmt="%.4g", padding=3)
        axis.set_title(name.replace("_", " ").title())
        axis.set_ylim(0, max(values) * 1.18 if max(values) else 1)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    load_project_env()
    parser = argparse.ArgumentParser(description="Paired Traditional vs representation-state adaptive RAG evaluation")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--static-k", type=int,
                        help="Pilot override only; confirmatory runs use the frozen manifest")
    parser.add_argument("--calibration-manifest", type=Path,
                        help="Required for non-pilot runs; must be a frozen training-only manifest")
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--skip-quality-judge", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--condition", choices=CONDITIONS, default="full")
    parser.add_argument("--resume", action="store_true",
                        help="Resume only when the saved run fingerprint matches exactly")
    parser.add_argument("--input-cost-per-million", type=float,
                        default=os.environ.get("GROQ_INPUT_COST_PER_MILLION"))
    parser.add_argument("--cached-input-cost-per-million", type=float,
                        default=os.environ.get("GROQ_CACHED_INPUT_COST_PER_MILLION"))
    parser.add_argument("--output-cost-per-million", type=float,
                        default=os.environ.get("GROQ_OUTPUT_COST_PER_MILLION"))
    args = parser.parse_args(argv)
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY is missing")
    model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
    frozen = None
    if args.calibration_manifest:
        calibration = json.loads(args.calibration_manifest.read_text(encoding="utf-8"))
        if calibration.get("status") != "training_only_calibration_frozen" or not calibration.get("frozen_configuration"):
            raise SystemExit("calibration manifest is not a frozen training-only configuration")
        frozen = calibration["frozen_configuration"]
    elif not args.pilot:
        raise SystemExit("non-pilot evaluation requires --calibration-manifest")
    if not args.pilot and args.limit is not None:
        raise SystemExit("confirmatory evaluation forbids --limit; evaluate the frozen input file in full")
    rate_values = (
        args.input_cost_per_million,
        args.cached_input_cost_per_million,
        args.output_cost_per_million,
    )
    if any(value is None for value in rate_values):
        if not args.pilot:
            raise SystemExit("confirmatory evaluation requires all three frozen Groq cost rates")
        cost_rates = None
    else:
        cost_rates = tuple(float(value) for value in rate_values)
        if any(value < 0 for value in cost_rates):
            raise SystemExit("Groq cost rates cannot be negative")
    proposed_config = replace(
        SarRagConfig(groq_model=model),
        **(frozen["proposed"] if frozen else {}),
    )
    static_k = int(frozen["baseline"]["static_k"]) if frozen else (args.static_k or 5)
    if frozen and args.static_k is not None and args.static_k != static_k:
        raise SystemExit("--static-k cannot override a frozen confirmatory manifest")
    proposed_config, static_k, proposed_engine = apply_condition(
        proposed_config, static_k, args.condition
    )
    rows = _load_jsonl(args.input)
    if args.limit is not None:
        rows = rows[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    example_ids = [row["example_id"] for row in rows]
    if len(example_ids) != len(set(example_ids)):
        raise SystemExit("evaluation input contains duplicate example_id values")
    run_fingerprint = {
        "input_sha256": sha256(args.input.read_bytes()).hexdigest(),
        "calibration_sha256": (
            sha256(args.calibration_manifest.read_bytes()).hexdigest()
            if args.calibration_manifest else None
        ),
        "model": model,
        "pilot": args.pilot,
        "limit": args.limit,
        "skip_quality_judge": args.skip_quality_judge,
        "seed": args.seed,
        "static_k": static_k,
        "condition": args.condition,
        "proposed_engine": proposed_engine,
        "cost_rates_per_million_usd": cost_rates,
        "proposed_config": {
            name: getattr(proposed_config, name)
            for name in proposed_config.__dataclass_fields__
        },
    }
    # Normalize tuples and other JSON-compatible values before equality checks.
    run_fingerprint = json.loads(json.dumps(run_fingerprint, sort_keys=True))
    run_manifest_path = args.output_dir / "run_fingerprint.json"
    checkpoint_dir = args.output_dir / "checkpoints"
    if run_manifest_path.exists():
        saved_fingerprint = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if not args.resume:
            raise SystemExit("output directory already contains a run; pass --resume or use a new directory")
        if saved_fingerprint != run_fingerprint:
            raise SystemExit("resume refused: saved run fingerprint does not match this invocation")
    else:
        if args.resume:
            raise SystemExit("resume requested but no run fingerprint exists")
        run_manifest_path.write_text(
            json.dumps(run_fingerprint, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    encoder = SentenceTransformerEncoder("models/all-MiniLM-L6-v2")
    proposed_client = GroqClient(model)
    baseline_client = GroqClient(model)
    evaluator_client = GroqClient(model)
    records = []
    if args.resume and checkpoint_dir.exists():
        for path in checkpoint_dir.glob("*.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            expected_name = sha256(record["example_id"].encode("utf-8")).hexdigest() + ".json"
            if path.name != expected_name:
                raise SystemExit(f"invalid checkpoint filename: {path.name}")
            records.append(record)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    completed_ids = {record["example_id"] for record in records}
    cached_corpus_key = None
    proposed: SarRagSystem | LlmJudgedSarRagSystem | None = None
    baseline: TraditionalRagSystem | None = None
    seen_corpus_hashes: set[str] = set()
    proposed_index_seconds = 0.0
    baseline_index_seconds = 0.0
    proposed_index_vector_bytes = 0
    baseline_index_vector_bytes = 0
    proposed_indexed_chunks = 0
    baseline_indexed_chunks = 0
    proposed_embedding_calls = 0
    baseline_embedding_calls = 0
    proposed_embedded_texts = 0
    baseline_embedded_texts = 0
    for index, row in enumerate(rows):
        if row["example_id"] in completed_ids:
            continue
        documents = _documents(row)
        corpus_key = json.dumps(row["documents"], sort_keys=True, ensure_ascii=False)
        corpus_hash = sha256(corpus_key.encode("utf-8")).hexdigest()
        seen_corpus_hashes.add(corpus_hash)
        if corpus_key != cached_corpus_key:
            calls_before = encoder.encode_calls
            texts_before = encoder.encoded_texts
            started = perf_counter()
            proposed_artifacts = build_corpus(
                documents, encoder, proposed_config.enabled_representations
            )
            if proposed_engine == "legacy_multi_call":
                proposed = LlmJudgedSarRagSystem(
                    proposed_artifacts,
                    GroqRequirementExtractor(proposed_client),
                    GroqEvidenceJudge(proposed_client),
                    GroqReformulator(proposed_client),
                    GroqAnswerGenerator(proposed_client),
                    proposed_config,
                )
            else:
                proposed = SarRagSystem(
                    proposed_artifacts, encoder, GroqOneCallAnswerGenerator(proposed_client),
                    proposed_config,
                )
            proposed_index_seconds += perf_counter() - started
            proposed_index_vector_bytes += proposed_artifacts.index_vector_bytes
            proposed_indexed_chunks += proposed_artifacts.indexed_chunks
            proposed_embedding_calls += encoder.encode_calls - calls_before
            proposed_embedded_texts += encoder.encoded_texts - texts_before
            calls_before = encoder.encode_calls
            texts_before = encoder.encoded_texts
            started = perf_counter()
            baseline = TraditionalRagSystem.build(
                documents, encoder, baseline_client,
                TraditionalRagConfig(static_k=static_k),
            )
            baseline_index_seconds += perf_counter() - started
            baseline_index_vector_bytes += baseline.artifacts.index_vector_bytes
            baseline_indexed_chunks += baseline.artifacts.indexed_chunks
            baseline_embedding_calls += encoder.encode_calls - calls_before
            baseline_embedded_texts += encoder.encoded_texts - texts_before
            cached_corpus_key = corpus_key
        assert proposed is not None and baseline is not None
        results, metrics = {}, {}
        order = ["traditional", "proposed"] if index % 2 == 0 else ["proposed", "traditional"]
        for name in order:
            client = baseline_client if name == "traditional" else proposed_client
            before = len(client.usage_log)
            attempts_before = len(client.attempt_log)
            started = perf_counter()
            result = baseline.ask(row["query"]) if name == "traditional" else proposed.ask(row["query"])
            wall = perf_counter() - started
            results[name] = result
            metrics[name] = method_metrics(
                result, row, wall, _usage(client.usage_log[before:], cost_rates)
            )
            metrics[name].update(_attempt_usage(client.attempt_log[attempts_before:]))
        evaluator_usage = {}
        if not args.skip_quality_judge:
            before = len(evaluator_client.usage_log)
            attempts_before = len(evaluator_client.attempt_log)
            quality = paired_quality_judge(evaluator_client, row["query"], results, bool(index % 2))
            evaluator_usage = _usage(evaluator_client.usage_log[before:], cost_rates)
            evaluator_usage.update(_attempt_usage(evaluator_client.attempt_log[attempts_before:]))
            for name in ("traditional", "proposed"):
                metrics[name].update(quality[name])
        records.append({
            "example_id": row["example_id"], "dataset": row.get("dataset"), "split": row.get("split"),
            "metadata": row.get("metadata", {}),
            "query": row["query"], "gold_answers": row["gold_answers"],
            "gold_evidence_spans": row.get("gold_evidence_spans", []),
            "gold_answerability": row.get("gold_answerability"),
            "gold_evidence_count": len(row.get("gold_evidence_spans", [])),
            "document_count": len(row.get("documents", [])),
            "document_words": sum(
                len(document.get("text", "").split()) for document in row.get("documents", [])
            ),
            "descriptors": _example_descriptors(row),
            "traditional": metrics["traditional"], "proposed": metrics["proposed"],
            "answers": {name: results[name].answer for name in results},
            "answerability": {name: results[name].answerability for name in results},
            "evidence": {
                name: [{
                    "candidate_id": item.candidate.candidate_id,
                    "document_id": item.candidate.document_id,
                    "representation": item.candidate.representation_id,
                    "rank": item.candidate.rank,
                    "text": item.chunk.evidence_text,
                    "provenance": [list(value) for value in item.provenance],
                } for item in results[name].final_evidence]
                for name in results
            },
            "provisional_evidence": {
                name: [{
                    "candidate_id": item.candidate.candidate_id,
                    "representation": item.candidate.representation_id,
                    "text": item.chunk.evidence_text,
                } for item in results[name].provisional_evidence]
                for name in results
            },
            "selected_representations": list(results["proposed"].selected_representations),
            "stop_reason": {
                name: results[name].stop_reason for name in ("traditional", "proposed")
            },
            "trace_summary": {
                name: _trace_summary(results[name]) for name in ("traditional", "proposed")
            },
            "representation_gold_diagnostic": _representation_gold_diagnostic(
                proposed, row["query"], tuple(row.get("gold_evidence_spans", [])),
                proposed_config.probe_depth,
            ),
            "evaluator_usage": evaluator_usage,
        })
        checkpoint_path = checkpoint_dir / (
            sha256(row["example_id"].encode("utf-8")).hexdigest() + ".json"
        )
        temporary_checkpoint = checkpoint_path.with_suffix(".json.tmp")
        temporary_checkpoint.write_text(
            json.dumps(records[-1], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary_checkpoint.replace(checkpoint_path)

    input_order = {example_id: index for index, example_id in enumerate(example_ids)}
    records.sort(key=lambda record: input_order[record["example_id"]])

    statistics = paired_statistics(records, args.bootstrap_resamples, args.seed)
    binary_metrics = [
        name for name in ("answer_exact_match", "complete_evidence_recovery", "answerability_correct")
        if any(name in record["traditional"] and name in record["proposed"] for record in records)
    ]
    mcnemar = {}
    for name in binary_metrics:
        pairs = [
            (record["traditional"].get(name), record["proposed"].get(name))
            for record in records
        ]
        pairs = [(a, b) for a, b in pairs if a is not None and b is not None
                 and not math.isnan(float(a)) and not math.isnan(float(b))]
        mcnemar[name] = exact_mcnemar_p_value(
            [int(a) for a, _ in pairs], [int(b) for _, b in pairs]
        )
    primary_metrics = [
        name for name in ("answer_token_f1", "evidence_set_recall",
                          "complete_evidence_recovery", "context_tokens")
        if name in statistics
    ]
    holm = holm_adjust({
        name: statistics[name]["paired_randomization_p_value"] for name in primary_metrics
    })
    summary = {
        "status": "pilot_not_for_claims" if args.pilot else "confirmatory",
        "examples": len(records), "model": model,
        "registered_condition": args.condition,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "traditional_condition": {"representation": "sliding_window", "static_k": static_k},
        "proposed_condition": {
            "representations": list(proposed_config.enabled_representations),
            "probe_depth": 3,
            "representation_selection": "normalized_rrf_greedy_weighted_set_cover",
            "final_k": "emergent",
            "online_llm_calls": "one final answer call",
        },
        "calibration_manifest": str(args.calibration_manifest) if args.calibration_manifest else None,
        "cost_rates_per_million_usd": {
            "uncached_input": cost_rates[0], "cached_input": cost_rates[1],
            "output": cost_rates[2],
        } if cost_rates is not None else None,
        "offline_indexing": {
            "unique_corpora_built_this_invocation": len(seen_corpus_hashes),
            "traditional_seconds": baseline_index_seconds,
            "proposed_seconds": proposed_index_seconds,
            "scope": "corpora built in this process invocation; excluded from per-query latency",
            "resume_invocation": args.resume,
            "cache_policy": "one active corpus; prevents benchmark-size index accumulation",
            "traditional_index_vector_bytes": baseline_index_vector_bytes,
            "proposed_index_vector_bytes": proposed_index_vector_bytes,
            "traditional_indexed_chunks": baseline_indexed_chunks,
            "proposed_indexed_chunks": proposed_indexed_chunks,
            "traditional_embedding_calls": baseline_embedding_calls,
            "proposed_embedding_calls": proposed_embedding_calls,
            "traditional_embedded_texts": baseline_embedded_texts,
            "proposed_embedded_texts": proposed_embedded_texts,
            "memory_caveat": "vector-array bytes are exact; process peak working set is OS-reported for the complete evaluator process and is not attributable to one method",
            "process_peak_working_set_bytes": _windows_peak_working_set_bytes(),
        },
        "api_attempts": {
            "traditional": dict(Counter(item["status"] for item in baseline_client.attempt_log)),
            "proposed": dict(Counter(item["status"] for item in proposed_client.attempt_log)),
            "evaluator": dict(Counter(item["status"] for item in evaluator_client.attempt_log)),
        },
        "answerability": {
            "traditional": classification_summary(records, "traditional"),
            "proposed": classification_summary(records, "proposed"),
        },
        "paired_statistics": statistics,
        "exact_mcnemar_tests": mcnemar,
        "holm_adjusted_primary_p_values": holm,
        "predeclared_practical_margins": {
            "answer_token_f1_superiority": 0.02,
            "evidence_set_recall_noninferiority": -0.02,
            "complete_evidence_recovery_noninferiority": -0.02,
            "context_token_reduction_fraction": 0.10,
            "cost_reduction_fraction": 0.10,
        },
        "metric_caveats": {
            "answer_generator_control": "both retrieval methods use the identical Groq prompt, schema, model, decoding controls, and maximum completion tokens",
            "faithfulness_answer_relevancy": "same Groq model family as generator; secondary and self-evaluation-biased",
            "retrieval_relevance": "normalized containment match between final evidence text and gold evidence spans",
            "rank_metrics": "chunk-level precision, MRR, and nDCG depend on representation granularity; gold-span coverage recall and complete-evidence recovery are primary retrieval outcomes",
            "cost_rates": "explicitly frozen in the run fingerprint; cost is undefined when rates are omitted in pilot mode",
        },
    }
    (args.output_dir / "results.json").write_text(json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "paired_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["example_id", "method"] + sorted(set(records[0]["traditional"]) | set(records[0]["proposed"])) if records else []
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for record in records:
            for method in ("traditional", "proposed"):
                writer.writerow({"example_id": record["example_id"], "method": method, **record[method]})
    save_graphs(statistics, args.output_dir / "comparison.png")
    print(json.dumps({"output": str(args.output_dir), "examples": len(records), "status": summary["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
