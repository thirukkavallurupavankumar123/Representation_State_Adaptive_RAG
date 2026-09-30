"""Run the frozen, free retrieval-only Adaptive Single-View V2 recalibration.

Only official training files are opened.  The command is resumable under an
exact experiment fingerprint and never imports a network or Groq client.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from hashlib import sha256
import importlib.metadata
from itertools import chain, product
import json
import math
from pathlib import Path
import platform
import sys
import time
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sar_rag.adaptive_pipeline import SarRagSystem  # noqa: E402
from sar_rag.adaptive_single_view_v2 import (  # noqa: E402
    AdaptiveSingleViewV2Config, AdaptiveSingleViewV2System, V2RetrievalResult,
    canonical_sha256, file_sha256, retrieval_metrics,
)
from sar_rag.benchmark import _documents  # noqa: E402
from sar_rag.calibration import (  # noqa: E402
    RetrievalOnlyAnswerer, _covered_spans, deterministic_stratified_sample,
    partition_training_sample, stratum,
)
from sar_rag.config import SarRagConfig  # noqa: E402
from sar_rag.corpus import CorpusArtifacts, build_corpus  # noqa: E402
from sar_rag.dataset_adapters import iter_musique_full, iter_qasper  # noqa: E402
from sar_rag.encoding import SentenceTransformerEncoder  # noqa: E402
from sar_rag.models import Candidate, EvidenceItem, Judgment  # noqa: E402

REPRESENTATIONS = ("semantic", "paragraph", "hierarchical_v2", "sentence_window")
REPRESENTATION_TIE_ORDER = {name: pos for pos, name in enumerate(REPRESENTATIONS)}
TOKEN_POINTS = (800, 1200, 1600)
TARGET_STRATA = {
    "qasper|ANSWERABLE|short", "qasper|ANSWERABLE|medium", "qasper|ANSWERABLE|long",
    "musique_full|ANSWERABLE|2hop", "musique_full|ANSWERABLE|3hop", "musique_full|ANSWERABLE|4hop",
    "musique_full|INSUFFICIENT|2hop", "musique_full|INSUFFICIENT|3hop", "musique_full|INSUFFICIENT|4hop",
}
GRID = {
    "relevance": (0.45, 0.55, 0.65, 0.75), "patience": (1, 2, 3),
    "rounds": (6, 8), "candidates": (30, 60), "chunks": (12, 20),
}
SEED = 20260905


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows and not fields:
        raise ValueError(f"fields required for empty CSV: {path}")
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def load_exclusions() -> tuple[set[str], dict]:
    path = ROOT / "output" / "adaptive_single_view_v2" / "DESIGN_EXPOSED_TRAIN_IDS.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = set(payload["ids"])
    if len(ids) != payload["count"] or any(":train:" not in value for value in ids):
        raise ValueError("invalid design-exposed training registry")
    return ids, payload


def fresh_sample() -> tuple[list[dict], dict[str, str], dict]:
    exclusions, exclusion_payload = load_exclusions()
    raw = ROOT / "datasets" / "raw"
    source = chain(
        iter_qasper(raw / "qasper" / "data" / "qasper-train-v0.3.json", "train"),
        iter_musique_full(raw / "musique" / "data" / "musique_full_v1.0_train.jsonl", "train"),
    )
    eligible = (row for row in source if row["example_id"] not in exclusions
                and row.get("gold_evidence_spans") and stratum(row) in TARGET_STRATA)
    sample, population = deterministic_stratified_sample(eligible, 25, SEED)
    counts = Counter(stratum(row) for row in sample)
    if set(counts) != TARGET_STRATA or set(counts.values()) != {25}:
        raise ValueError(f"sample failed frozen strata/counts: {counts}")
    partitions = partition_training_sample(sample, SEED, 0.28)
    partition_counts = Counter(partitions.values())
    if partition_counts != {"tuning": 162, "selection": 63}:
        raise ValueError(f"partition failed frozen counts: {partition_counts}")
    sample_ids = {row["example_id"] for row in sample}
    heldout = json.loads((ROOT / "output" / "adaptive_single_view_v2" /
                          "HELDOUT_ID_REGISTRY.json").read_text(encoding="utf-8"))
    heldout_ids = set(heldout["ids"])
    audit = {
        "seed": SEED, "sample_count": len(sample), "stratum_counts": dict(sorted(counts.items())),
        "partition_counts": dict(partition_counts), "population_after_exclusion": population,
        "design_exposed_registry_sha256": exclusion_payload["ids_sha256"],
        "design_exposed_overlap": len(sample_ids & exclusions),
        "heldout_id_registry_sha256": heldout["ids_sha256"],
        "heldout_id_overlap": len(sample_ids & heldout_ids),
        "all_ids_are_training": all(":train:" in value for value in sample_ids),
        "qasper_test_overlap_structural": sum(value.startswith("qasper:test:") for value in sample_ids),
    }
    if any((audit["design_exposed_overlap"], audit["heldout_id_overlap"],
            audit["qasper_test_overlap_structural"])) or not audit["all_ids_are_training"]:
        raise ValueError(f"sample overlap audit failed: {audit}")
    return sample, partitions, audit


def restrict(artifacts: CorpusArtifacts, representation: str) -> CorpusArtifacts:
    index = artifacts.indexes[representation]
    return CorpusArtifacts(artifacts.units_by_id,
                           {chunk.chunk_id: chunk for chunk in index.chunks},
                           {representation: index})


def config_id(config: AdaptiveSingleViewV2Config, mode: str = "adaptive") -> str:
    return f"{mode}-{config.fingerprint[:20]}"


def metric_fields(result: V2RetrievalResult, row: dict) -> dict:
    metric = retrieval_metrics(result, row)
    return {
        "evidence_recall": metric["evidence_recall"],
        "evidence_precision": metric["evidence_precision"],
        "complete_evidence": metric["complete_evidence"],
        "mrr": metric["mrr"], "ndcg": metric["ndcg"],
        "final_k": result.final_k, "pre_dedup_k": result.pre_dedup_k,
        "post_dedup_k": result.post_dedup_k, "context_tokens": result.context_tokens,
        "candidates_examined": result.candidates_examined,
        "retrieval_rounds": result.retrieval_rounds, "stop_reason": result.stop_reason,
    }


FIELDS = [
    "experiment_fingerprint", "example_id", "dataset", "stratum", "answerable",
    "design_partition", "mode", "configuration_id", "configuration_fingerprint",
    "representation", "token_point", "relevance_threshold", "patience", "max_rounds",
    "max_candidates", "max_chunks", "evidence_recall", "evidence_precision",
    "complete_evidence", "mrr", "ndcg", "final_k", "pre_dedup_k", "post_dedup_k",
    "context_tokens", "candidates_examined", "retrieval_rounds", "stop_reason", "latency_ms",
]


def row_record(experiment_fingerprint: str, row: dict, partition: str, mode: str,
               config: AdaptiveSingleViewV2Config, result: V2RetrievalResult,
               elapsed_ms: float) -> dict:
    return {
        "experiment_fingerprint": experiment_fingerprint, "example_id": row["example_id"],
        "dataset": row["dataset"], "stratum": stratum(row),
        "answerable": int(row.get("gold_answerability") == "ANSWERABLE"),
        "design_partition": partition, "mode": mode,
        "configuration_id": config_id(config, mode),
        "configuration_fingerprint": config.fingerprint,
        "representation": config.representation, "token_point": config.max_context_tokens,
        "relevance_threshold": config.relevance_percentile_threshold,
        "patience": config.empty_batch_patience, "max_rounds": config.max_retrieval_rounds,
        "max_candidates": config.max_candidates_examined, "max_chunks": config.max_final_chunks,
        **metric_fields(result, row), "latency_ms": elapsed_ms,
    }


def fixed_result(artifacts: CorpusArtifacts, encoder, config: AdaptiveSingleViewV2Config,
                 query: str, k: int) -> V2RetrievalResult:
    index = artifacts.indexes[config.representation]
    candidates = index.search(query, 1, k, "probe")
    accepted: list[EvidenceItem] = []
    used = 0
    for candidate in candidates:
        chunk = artifacts.chunks_by_id[candidate.chunk_id]
        tokens = encoder.token_count(chunk.evidence_text)
        if used + tokens > config.max_context_tokens or len(accepted) >= config.max_final_chunks:
            continue
        accepted.append(EvidenceItem(candidate, chunk, Judgment(candidate.candidate_id, ()),
                                     {f"UNIT:{unit}" for unit in candidate.canonical_unit_ids},
                                     [(candidate.representation_id, candidate.rank)]))
        used += tokens
    trace = [{"event": "fixed_k_retrieval_completed_without_llm", "k": k, "llm_calls": 0}]
    return V2RetrievalResult(
        query, config.representation, accepted, list(accepted), len(candidates), 1,
        len(accepted), len(accepted), len(accepted), used, f"fixed_k{k}",
        {"max_context_tokens": config.max_context_tokens, "fixed_k": k},
        config.fingerprint, "measured-in-run-manifest", trace,
    )


def configurations() -> list[AdaptiveSingleViewV2Config]:
    return [AdaptiveSingleViewV2Config(
        representation=representation, max_context_tokens=tokens,
        relevance_percentile_threshold=relevance, empty_batch_patience=patience,
        max_retrieval_rounds=rounds, max_candidates_examined=candidates,
        max_final_chunks=chunks, seed=SEED,
    ) for representation, tokens, relevance, patience, rounds, candidates, chunks in product(
        REPRESENTATIONS, TOKEN_POINTS, GRID["relevance"], GRID["patience"], GRID["rounds"],
        GRID["candidates"], GRID["chunks"])]


def mean_defined(rows: list[dict], field: str, predicate=lambda row: True) -> float:
    values = [float(row[field]) for row in rows if predicate(row) and row[field] not in ("", None)
              and not math.isnan(float(row[field]))]
    return float(np.mean(values)) if values else float("nan")


def aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["configuration_id"]].append(row)
    output = []
    for identifier, values in grouped.items():
        first = values[0]
        output.append({
            "configuration_id": identifier, "mode": first["mode"],
            "configuration_fingerprint": first["configuration_fingerprint"],
            "representation": first["representation"], "token_point": int(first["token_point"]),
            "relevance_threshold": float(first["relevance_threshold"]),
            "patience": int(first["patience"]), "max_rounds": int(first["max_rounds"]),
            "max_candidates": int(first["max_candidates"]), "max_chunks": int(first["max_chunks"]),
            "n": len(values), "evidence_recall": mean_defined(values, "evidence_recall"),
            "answerable_CER": mean_defined(values, "complete_evidence", lambda row: int(row["answerable"]) == 1),
            "insufficient_residual_support_complete_recovery": mean_defined(
                values, "complete_evidence", lambda row: int(row["answerable"]) == 0),
            "evidence_precision": mean_defined(values, "evidence_precision"),
            "mrr": mean_defined(values, "mrr"), "ndcg": mean_defined(values, "ndcg"),
            "context_tokens": mean_defined(values, "context_tokens"),
            "final_k": mean_defined(values, "final_k"),
            "candidates_examined": mean_defined(values, "candidates_examined"),
            "retrieval_rounds": mean_defined(values, "retrieval_rounds"),
            "latency_ms": mean_defined(values, "latency_ms"),
        })
    return output


def tuning_select(aggregates: list[dict], configs: list[AdaptiveSingleViewV2Config]) -> list[dict]:
    by_id = {config_id(config): config for config in configs}
    selected = []
    for representation in REPRESENTATIONS:
        for token_point in TOKEN_POINTS:
            pool = [row for row in aggregates if row["representation"] == representation
                    and int(row["token_point"]) == token_point and row["mode"] == "adaptive"]
            best_recall = max(row["evidence_recall"] for row in pool)
            shortlist = [row for row in pool if row["evidence_recall"] >= best_recall - 0.01]
            winner = min(shortlist, key=lambda row: (
                -row["answerable_CER"], row["context_tokens"], row["candidates_examined"],
                float(row["relevance_threshold"]), int(row["patience"]), int(row["max_rounds"]),
                int(row["max_candidates"]), int(row["max_chunks"]),
            ))
            selected.append({**winner, "configuration": asdict(by_id[winner["configuration_id"]])})
    return selected


def paired(reference: list[float], condition: list[float], seed: int = SEED,
           resamples: int = 20_000) -> dict:
    pairs = [(a, b) for a, b in zip(reference, condition)
             if not math.isnan(a) and not math.isnan(b)]
    delta = np.asarray([b - a for a, b in pairs], dtype=float)
    rng = np.random.default_rng(seed)
    bootstrap = delta[rng.integers(0, len(delta), size=(resamples, len(delta)))].mean(axis=1)
    signs = rng.integers(0, 2, size=(resamples, len(delta))) * 2 - 1
    randomised = (delta * signs).mean(axis=1)
    observed = float(delta.mean())
    return {"n": len(delta), "mean_reference": float(np.mean([a for a, _ in pairs])),
            "mean_condition": float(np.mean([b for _, b in pairs])), "difference": observed,
            "ci_low": float(np.quantile(bootstrap, .025)),
            "ci_high": float(np.quantile(bootstrap, .975)),
            "p_value": float((np.sum(np.abs(randomised) >= abs(observed)) + 1) / (resamples + 1))}


def mcnemar(reference: list[float], condition: list[float]) -> dict:
    pairs = [(int(a), int(b)) for a, b in zip(reference, condition)
             if not math.isnan(a) and not math.isnan(b)]
    ref_only = sum(a == 1 and b == 0 for a, b in pairs)
    cond_only = sum(a == 0 and b == 1 for a, b in pairs)
    discordant = ref_only + cond_only
    tail = min(ref_only, cond_only)
    p = 1.0 if not discordant else min(
        1.0, 2 * sum(math.comb(discordant, k) for k in range(tail + 1)) / 2**discordant)
    return {"reference_only": ref_only, "condition_only": cond_only,
            "discordant": discordant, "p_value": p}


def holm(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: item[1])
    output, running = {}, 0.0
    for rank, (name, value) in enumerate(ordered, 1):
        running = max(running, min(1.0, (len(ordered) - rank + 1) * value))
        output[name] = running
    return output


def paired_analysis(selection_rows: list[dict], winner_id: str) -> list[dict]:
    winner_rows = {row["example_id"]: row for row in selection_rows
                   if row["configuration_id"] == winner_id}
    winner = next(iter(winner_rows.values()))
    comparator_modes = ("probe_only", "fixed_k10", "fixed_k20")
    comparators = {}
    for mode in comparator_modes:
        matches = [row for row in selection_rows if row["mode"] == mode
                   and row["representation"] == winner["representation"]
                   and int(row["token_point"]) == int(winner["token_point"])]
        comparators[mode] = {row["example_id"]: row for row in matches}
    results, p_families = [], defaultdict(dict)
    metric_specs = (
        ("evidence_recall", lambda row: True),
        ("answerable_CER", lambda row: int(row["answerable"]) == 1),
        ("context_tokens", lambda row: True),
    )
    for metric, predicate in metric_specs:
        field = "complete_evidence" if metric == "answerable_CER" else metric
        for mode, values in comparators.items():
            ids = sorted(set(winner_rows) & set(values))
            ids = [example_id for example_id in ids if predicate(winner_rows[example_id])]
            # Difference is winner minus comparator.
            stats = paired([float(values[i][field]) for i in ids],
                           [float(winner_rows[i][field]) for i in ids])
            row = {"metric": metric, "reference": mode, "condition": "selected_adaptive", **stats}
            if metric == "answerable_CER":
                row.update({f"mcnemar_{key}": value for key, value in
                            mcnemar([float(values[i][field]) for i in ids],
                                    [float(winner_rows[i][field]) for i in ids]).items()})
            results.append(row)
            p_families[metric][mode] = stats["p_value"]
    adjusted = {metric: holm(values) for metric, values in p_families.items()}
    for row in results:
        row["holm_p"] = adjusted[row["metric"]][row["reference"]]
        row["holm_family_size"] = len(adjusted[row["metric"]])
    return results


def choose_final(selection_aggregates: list[dict], tuning_selected: list[dict]) -> dict:
    eligible_ids = {row["configuration_id"] for row in tuning_selected}
    candidates = [row for row in selection_aggregates if row["configuration_id"] in eligible_ids]
    best_recall = max(row["evidence_recall"] for row in candidates)
    shortlist = [row for row in candidates if row["evidence_recall"] >= best_recall - 0.01]
    return min(shortlist, key=lambda row: (
        -row["answerable_CER"], row["context_tokens"], row["candidates_examined"],
        REPRESENTATION_TIE_ORDER[row["representation"]], int(row["token_point"]),
    ))


def plot_artifacts(out: Path, selection_aggregates: list[dict], winner: dict,
                   winner_rows: list[dict]) -> None:
    plt.figure(figsize=(8, 5))
    for mode, marker in (("adaptive", "o"), ("probe_only", "x"),
                         ("fixed_k10", "s"), ("fixed_k20", "^"), ("v1_full", "D")):
        rows = [row for row in selection_aggregates if row["mode"] == mode]
        if rows:
            plt.scatter([row["context_tokens"] for row in rows],
                        [row["evidence_recall"] for row in rows], label=mode, marker=marker, alpha=.8)
    plt.scatter([winner["context_tokens"]], [winner["evidence_recall"]], s=180,
                facecolors="none", edgecolors="black", linewidths=1.5, label="selected")
    plt.xlabel("Mean actual context tokens"); plt.ylabel("Unique gold-span recall")
    plt.title("Adaptive Single-View V2: selection recall-token trade-off")
    plt.grid(alpha=.2); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(out / "RECALL_TOKEN_PARETO.png", dpi=180); plt.close()

    plt.figure(figsize=(7, 4.5))
    ks = [int(row["final_k"]) for row in winner_rows]
    plt.hist(ks, bins=range(min(ks), max(ks) + 2), align="left", rwidth=.85)
    plt.xlabel("Final k (emergent)"); plt.ylabel("Questions"); plt.title("Selected v2 final-k distribution")
    plt.tight_layout(); plt.savefig(out / "FINAL_K_HISTOGRAM.png", dpi=180); plt.close()

    counts = Counter(row["stop_reason"] for row in winner_rows)
    plt.figure(figsize=(8, 4.5))
    names = list(counts); values = [counts[name] for name in names]
    plt.bar(names, values); plt.ylabel("Questions"); plt.title("Selected v2 stop reasons")
    plt.xticks(rotation=25, ha="right"); plt.tight_layout()
    plt.savefig(out / "STOP_REASON_DISTRIBUTION.png", dpi=180); plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "output" / "adaptive_single_view_v2" / "recalibration_seed20260905")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    out = args.output_dir
    protocol_dir = ROOT / "output" / "adaptive_single_view_v2"
    protocol = protocol_dir / "PROTOCOL.json"
    amendment = protocol_dir / "PROTOCOL_AMENDMENT_001.json"
    sample, partitions, overlap_audit = fresh_sample()
    configs = configurations()
    sample_identity = [{"example_id": row["example_id"], "partition": partitions[row["example_id"]]}
                       for row in sample]
    experiment_spec = {
        "method": "adaptive_single_view_v2_recalibration", "seed": SEED,
        "protocol_sha256": file_sha256(protocol), "amendment_sha256": file_sha256(amendment),
        "code_sha256": file_sha256(Path(__file__)),
        "retriever_code_sha256": file_sha256(ROOT / "src" / "sar_rag" / "adaptive_single_view_v2.py"),
        "sample_sha256": canonical_sha256(sample_identity),
        "config_fingerprints": sorted(config.fingerprint for config in configs),
        "raw_inputs": {
            "qasper_train": file_sha256(ROOT / "datasets/raw/qasper/data/qasper-train-v0.3.json"),
            "musique_train": file_sha256(ROOT / "datasets/raw/musique/data/musique_full_v1.0_train.jsonl"),
        },
    }
    experiment_fingerprint = canonical_sha256(experiment_spec)
    run_manifest = {
        "status": "RUNNING_RETRIEVAL_ONLY", "experiment_fingerprint": experiment_fingerprint,
        "experiment_spec": experiment_spec, "overlap_audit": overlap_audit,
        "exact_command": ".venv\\Scripts\\python.exe tools\\run_adaptive_single_view_v2_recalibration.py --output-dir output\\adaptive_single_view_v2\\recalibration_seed20260905",
        "python": sys.version, "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("numpy", "torch", "transformers", "matplotlib", "pytest")},
        "llm_calls": 0, "failures": [], "retries": 0,
    }
    manifest_path = out / "RUN_MANIFEST.json"
    tuning_path = out / "TUNING_PER_EXAMPLE.csv"
    selection_path = out / "SELECTION_PER_EXAMPLE.csv"
    if out.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite existing run: {out}")
    out.mkdir(parents=True, exist_ok=True)
    if args.resume:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing["experiment_fingerprint"] != experiment_fingerprint:
            raise ValueError("resume experiment fingerprint mismatch")
        run_manifest = existing
        run_manifest["retries"] = int(run_manifest.get("retries", 0)) + 1
    else:
        manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
        with (out / "SAMPLE.jsonl").open("w", encoding="utf-8") as stream:
            for row in sample:
                stream.write(json.dumps({**row, "design_partition": partitions[row["example_id"]]},
                                        ensure_ascii=False) + "\n")
        (out / "SAMPLE_MANIFEST.json").write_text(json.dumps({
            **overlap_audit, "sample_ids": sample_identity,
            "sample_file_sha256": file_sha256(out / "SAMPLE.jsonl"),
        }, indent=2) + "\n", encoding="utf-8")

    existing_tuning = set()
    if tuning_path.exists():
        with tuning_path.open(encoding="utf-8", newline="") as stream:
            prior = list(csv.DictReader(stream))
        if any(row["experiment_fingerprint"] != experiment_fingerprint for row in prior):
            raise ValueError("tuning output contains another experiment fingerprint")
        existing_tuning = {(row["example_id"], row["configuration_id"]) for row in prior}

    encoder = SentenceTransformerEncoder(str(ROOT / "models" / "all-MiniLM-L6-v2"))
    model_weight = ROOT / "models" / "all-MiniLM-L6-v2" / "model.safetensors"
    embedding_payload = {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "dimension": encoder.dimension, "weight_sha256": file_sha256(model_weight),
    }
    embedding_fingerprint = canonical_sha256(embedding_payload)
    run_manifest["embedding"] = {**embedding_payload, "fingerprint": embedding_fingerprint}
    manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")

    corpus_cache: dict[str, CorpusArtifacts] = {}
    tuning = [row for row in sample if partitions[row["example_id"]] == "tuning"]
    selection = [row for row in sample if partitions[row["example_id"]] == "selection"]
    all_views = (*REPRESENTATIONS, "hierarchical")

    def artifacts_for(row: dict) -> CorpusArtifacts:
        docs = _documents(row)
        key = canonical_sha256([asdict(document) for document in docs])
        if key not in corpus_cache:
            corpus_cache[key] = build_corpus(docs, encoder, all_views)
        return corpus_cache[key]

    for position, row in enumerate(tuning, 1):
        try:
            art = artifacts_for(row)
            records = []
            for config in configs:
                identifier = config_id(config)
                if (row["example_id"], identifier) in existing_tuning:
                    continue
                started = time.perf_counter()
                result = AdaptiveSingleViewV2System(
                    restrict(art, config.representation), encoder, config,
                    embedding_fingerprint).retrieve(row["query"])
                records.append(row_record(experiment_fingerprint, row, "tuning", "adaptive",
                                          config, result, (time.perf_counter() - started) * 1000))
            if records:
                append_csv(tuning_path, records, FIELDS)
            print(json.dumps({"stage": "tuning", "completed": position, "total": len(tuning),
                              "example_id": row["example_id"]}), flush=True)
        except Exception as exc:
            failure = {"stage": "tuning", "example_id": row["example_id"],
                       "type": type(exc).__name__, "message": str(exc)}
            run_manifest["failures"].append(failure)
            manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
            raise

    with tuning_path.open(encoding="utf-8", newline="") as stream:
        tuning_rows = list(csv.DictReader(stream))
    expected_tuning_rows = len(tuning) * len(configs)
    if len(tuning_rows) != expected_tuning_rows:
        raise ValueError(f"tuning row count {len(tuning_rows)} != {expected_tuning_rows}")
    tuning_aggregate = aggregate(tuning_rows)
    write_csv(out / "TUNING_AGGREGATE.csv", tuning_aggregate)
    selected_configs = tuning_select(tuning_aggregate, configs)
    (out / "TUNING_SELECTED_CONFIGS.json").write_text(
        json.dumps(selected_configs, indent=2) + "\n", encoding="utf-8")
    chosen_configs = [AdaptiveSingleViewV2Config(**row["configuration"]) for row in selected_configs]

    existing_selection = set()
    if selection_path.exists():
        with selection_path.open(encoding="utf-8", newline="") as stream:
            prior = list(csv.DictReader(stream))
        if any(row["experiment_fingerprint"] != experiment_fingerprint for row in prior):
            raise ValueError("selection output contains another experiment fingerprint")
        existing_selection = {(row["example_id"], row["configuration_id"]) for row in prior}

    v1_manifest = json.loads((ROOT / "output" / "calibration" / "frozen_manifest.json").read_text(encoding="utf-8"))
    v1_config = SarRagConfig(**v1_manifest["frozen_configuration"]["proposed"])
    for position, row in enumerate(selection, 1):
        try:
            art = artifacts_for(row)
            records = []
            for config in chosen_configs:
                modes = [("adaptive", None), ("probe_only", None), ("fixed_k10", 10), ("fixed_k20", 20)]
                for mode, fixed_k in modes:
                    effective = config if mode != "probe_only" else replace(config, adaptive_retrieval_enabled=False)
                    identifier = config_id(effective, mode)
                    if (row["example_id"], identifier) in existing_selection:
                        continue
                    started = time.perf_counter()
                    single_art = restrict(art, effective.representation)
                    result = (fixed_result(single_art, encoder, effective, row["query"], fixed_k)
                              if fixed_k else AdaptiveSingleViewV2System(
                                  single_art, encoder, effective, embedding_fingerprint).retrieve(row["query"]))
                    record = row_record(experiment_fingerprint, row, "selection", mode,
                                        effective, result, (time.perf_counter() - started) * 1000)
                    record["configuration_id"] = identifier
                    records.append(record)
            v1_identifier = "v1_full_frozen"
            if (row["example_id"], v1_identifier) not in existing_selection:
                started = time.perf_counter()
                v1_views = v1_config.enabled_representations
                v1_art = CorpusArtifacts(art.units_by_id,
                    {chunk.chunk_id: chunk for view in v1_views for chunk in art.indexes[view].chunks},
                    {view: art.indexes[view] for view in v1_views})
                v1_result = SarRagSystem(v1_art, encoder, RetrievalOnlyAnswerer(), v1_config).ask(row["query"])
                done = next(event for event in reversed(v1_result.trace) if event.get("event") == "run_completed")
                gold = tuple(row["gold_evidence_spans"])
                covered, relevant = _covered_spans(v1_result.final_evidence, gold)
                records.append({
                    "experiment_fingerprint": experiment_fingerprint, "example_id": row["example_id"],
                    "dataset": row["dataset"], "stratum": stratum(row),
                    "answerable": int(row.get("gold_answerability") == "ANSWERABLE"),
                    "design_partition": "selection", "mode": "v1_full",
                    "configuration_id": v1_identifier,
                    "configuration_fingerprint": canonical_sha256(asdict(v1_config)),
                    "representation": "v1_routed", "token_point": 2000,
                    "relevance_threshold": v1_config.similarity_threshold,
                    "patience": v1_config.empty_batch_patience, "max_rounds": v1_config.max_retrieval_rounds,
                    "max_candidates": v1_config.max_candidates_examined,
                    "max_chunks": v1_config.max_final_chunks,
                    "evidence_recall": len(covered) / len(gold),
                    "evidence_precision": sum(relevant) / len(relevant) if relevant else 0.0,
                    "complete_evidence": float(len(covered) == len(gold)), "mrr": "", "ndcg": "",
                    "final_k": v1_result.final_k,
                    "pre_dedup_k": done.get("retrieved_before_deduplication", ""),
                    "post_dedup_k": v1_result.final_k,
                    "context_tokens": done["final_context_tokens"],
                    "candidates_examined": done["candidates_examined"],
                    "retrieval_rounds": done["retrieval_rounds"],
                    "stop_reason": v1_result.stop_reason,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                })
            if records:
                append_csv(selection_path, records, FIELDS)
            print(json.dumps({"stage": "selection", "completed": position,
                              "total": len(selection), "example_id": row["example_id"]}), flush=True)
        except Exception as exc:
            failure = {"stage": "selection", "example_id": row["example_id"],
                       "type": type(exc).__name__, "message": str(exc)}
            run_manifest["failures"].append(failure)
            manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
            raise

    with selection_path.open(encoding="utf-8", newline="") as stream:
        selection_rows = list(csv.DictReader(stream))
    expected_selection_rows = len(selection) * (len(chosen_configs) * 4 + 1)
    if len(selection_rows) != expected_selection_rows:
        raise ValueError(f"selection row count {len(selection_rows)} != {expected_selection_rows}")
    selection_aggregate = aggregate(selection_rows)
    write_csv(out / "SELECTION_AGGREGATE.csv", selection_aggregate)
    winner = choose_final(selection_aggregate, selected_configs)
    uncertainty = paired_analysis(selection_rows, winner["configuration_id"])
    write_csv(out / "PAIRED_UNCERTAINTY.csv", uncertainty)

    winner_rows = [row for row in selection_rows if row["configuration_id"] == winner["configuration_id"]]
    subgroups = []
    for group_type, names in (("dataset", sorted({row["dataset"] for row in winner_rows})),
                              ("stratum", sorted({row["stratum"] for row in winner_rows})),
                              ("answerability", ["answerable", "insufficient"])):
        for name in names:
            if group_type == "answerability":
                rows = [row for row in winner_rows if int(row["answerable"]) == (name == "answerable")]
            else:
                rows = [row for row in winner_rows if row[group_type] == name]
            subgroups.append({"group_type": group_type, "group": name, "n": len(rows),
                              "evidence_recall": mean_defined(rows, "evidence_recall"),
                              "complete_evidence": mean_defined(rows, "complete_evidence"),
                              "context_tokens": mean_defined(rows, "context_tokens"),
                              "final_k": mean_defined(rows, "final_k")})
    write_csv(out / "SUBGROUP_RESULTS.csv", subgroups)

    failures = [{"example_id": row["example_id"], "dataset": row["dataset"],
                 "stratum": row["stratum"], "answerable": row["answerable"],
                 "evidence_recall": row["evidence_recall"], "context_tokens": row["context_tokens"],
                 "final_k": row["final_k"], "stop_reason": row["stop_reason"],
                 "failure_class": "total_miss" if float(row["evidence_recall"]) == 0 else "partial"}
                for row in winner_rows if float(row["evidence_recall"]) < 1]
    write_csv(out / "FAILURE_ANALYSIS.csv", failures,
              ["example_id", "dataset", "stratum", "answerable", "evidence_recall",
               "context_tokens", "final_k", "stop_reason", "failure_class"])
    plot_artifacts(out, selection_aggregate, winner, winner_rows)

    comparison = {row["reference"]: row for row in uncertainty if row["metric"] == "evidence_recall"}
    probe = comparison["probe_only"]
    fixed20 = comparison["fixed_k20"]
    context_fixed20 = next(row for row in uncertainty if row["metric"] == "context_tokens"
                           and row["reference"] == "fixed_k20")
    cer_probe = next(row for row in uncertainty if row["metric"] == "answerable_CER"
                     and row["reference"] == "probe_only")
    quality = ((probe["difference"] >= 0.02 and probe["holm_p"] < 0.05)
               or (cer_probe["difference"] >= 0.03 and cer_probe["holm_p"] < 0.05))
    reduction_pct = (-100 * context_fixed20["difference"] / context_fixed20["mean_reference"]
                     if context_fixed20["mean_reference"] else 0.0)
    efficiency = fixed20["ci_low"] >= -0.02 and reduction_pct >= 10 and context_fixed20["holm_p"] < 0.05
    if quality:
        classification = "better_on_retrieval_quality"
    elif efficiency:
        classification = "better_only_on_efficiency"
    else:
        points = [(row["context_tokens"], row["evidence_recall"]) for row in selection_aggregate]
        dominated = any(tokens <= winner["context_tokens"] and recall >= winner["evidence_recall"]
                        and (tokens < winner["context_tokens"] or recall > winner["evidence_recall"])
                        for tokens, recall in points)
        classification = "unsupported" if dominated else "pareto_competitive"

    frozen_manifest = {
        "status": "FROZEN_RETRIEVAL_ONLY_V2", "experiment_fingerprint": experiment_fingerprint,
        "protocol_sha256": experiment_spec["protocol_sha256"],
        "protocol_amendment_sha256": experiment_spec["amendment_sha256"],
        "sample_sha256": experiment_spec["sample_sha256"],
        "embedding": run_manifest["embedding"], "selection_rule": "PROTOCOL.json frozen_selection_rule",
        "selected_configuration_id": winner["configuration_id"],
        "selected_configuration_fingerprint": winner["configuration_fingerprint"],
        "selected_configuration": next(row["configuration"] for row in selected_configs
                                       if row["configuration_id"] == winner["configuration_id"]),
        "selection_metrics": winner, "classification": classification,
        "paid_or_llm_calls": 0,
    }
    frozen_path = out / "FROZEN_V2_MANIFEST.json"
    frozen_path.write_text(json.dumps(frozen_manifest, indent=2) + "\n", encoding="utf-8")

    # Exact deterministic reproduction on the selection partition.  Corpora are
    # reused, but every retrieval decision is executed again from the manifest.
    selected_config = AdaptiveSingleViewV2Config(**frozen_manifest["selected_configuration"])
    reproduction = []
    for row in selection:
        result = AdaptiveSingleViewV2System(
            restrict(artifacts_for(row), selected_config.representation), encoder,
            selected_config, embedding_fingerprint).retrieve(row["query"])
        reproduction.append(row_record(experiment_fingerprint, row, "selection", "reproduction",
                                         selected_config, result, 0.0))
    write_csv(out / "REPRODUCTION_PER_EXAMPLE.csv", reproduction, FIELDS)
    original_signature = canonical_sha256([{key: row[key] for key in FIELDS if key != "latency_ms"}
                                           for row in winner_rows])
    reproduced_signature = canonical_sha256([{key: row[key] for key in FIELDS if key != "latency_ms"}
                                             for row in reproduction])
    reproduction_report = {
        "selected_manifest_sha256": file_sha256(frozen_path),
        "original_signature": original_signature,
        "reproduced_signature": reproduced_signature,
        "identical": original_signature == reproduced_signature,
    }
    # Mode and configuration_id are intentionally different labels; compare the
    # substantive deterministic result fields separately.
    substantive = ["example_id", "representation", "token_point", "evidence_recall",
                   "evidence_precision", "complete_evidence", "mrr", "ndcg", "final_k",
                   "pre_dedup_k", "post_dedup_k", "context_tokens", "candidates_examined",
                   "retrieval_rounds", "stop_reason", "configuration_fingerprint"]
    original_substantive = canonical_sha256([
        {key: str(row[key]) for key in substantive} for row in winner_rows])
    reproduced_substantive = canonical_sha256([
        {key: str(row[key]) for key in substantive} for row in reproduction])
    reproduction_report.update({"substantive_original_signature": original_substantive,
                                "substantive_reproduced_signature": reproduced_substantive,
                                "substantive_identical": original_substantive == reproduced_substantive})
    (out / "REPRODUCTION_VALIDATION.json").write_text(
        json.dumps(reproduction_report, indent=2) + "\n", encoding="utf-8")
    if not reproduction_report["substantive_identical"]:
        raise ValueError("frozen v2 manifest reproduction mismatch")

    run_manifest["status"] = "COMPLETE_RETRIEVAL_ONLY"
    run_manifest["llm_calls"] = 0
    run_manifest["selected_configuration_id"] = winner["configuration_id"]
    run_manifest["classification"] = classification
    artifacts = [path for path in out.iterdir() if path.is_file()]
    run_manifest["artifact_sha256"] = {path.name: file_sha256(path) for path in artifacts
                                       if path.name != "RUN_MANIFEST.json"}
    manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": run_manifest["status"], "selection": winner,
                      "classification": classification, "llm_calls": 0,
                      "reproduction_identical": True}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
