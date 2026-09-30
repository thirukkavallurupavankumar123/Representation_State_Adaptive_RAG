"""Post-seal gold acquisition, scoring and reporting for the reduced local-Qwen study.

Every subcommand first verifies PREDICTION_SEAL.json; nothing here runs before the seal.

  fetch-gold   download official QASPER test / MuSiQue-Full archives, verify SHA-256,
               regenerate the normalized files and prove they are the files the
               eligible pool was built from
  score        per-pair metrics, per-system aggregates, paired cluster inference + Holm
  report       final Markdown report and reproduction commands
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import importlib.util
import io
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from sar_rag.final_evaluation import file_sha256, object_sha256, token_f1  # noqa: E402
from sar_rag import local_qwen_4bit as lq  # noqa: E402
import local_qwen_4bit as gen  # noqa: E402


def _load_frozen_scorer():
    spec = importlib.util.spec_from_file_location("score_final_heldout_v1_1", ROOT / "tools" / "score_final_heldout_v1_1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FROZEN = _load_frozen_scorer()
OUT = gen.OUT
RAW = ROOT / "datasets" / "raw"
NORMALIZED = {"qasper": ROOT / "datasets" / "normalized" / "qasper_test.jsonl",
              "musique_full": ROOT / "datasets" / "normalized" / "musique_full_validation.jsonl"}
GOLD_PROVENANCE = OUT / "manifests" / "GOLD_PROVENANCE.json"
METRICS = OUT / "metrics"
COMPARISONS = OUT / "comparisons"
COMPARISON_PAIRS = [("adaptive_single_view_v2", "sar_rag_v1"), ("sar_rag_v1", "traditional_k10"),
                    ("adaptive_single_view_v2", "traditional_k10")]
PRIMARY = ["answer_token_f1", "supplied_evidence_recall", "answerable_complete_evidence", "supplied_context_tokens"]
SECONDARY = ["answer_exact_match", "answerability_correct", "abstained", "correct_abstention", "incorrect_abstention",
             "false_answer_on_insufficient", "insufficient_residual_support_recovery", "native_evidence_recall",
             "valid_generation", "input_tokens", "output_tokens", "qwen_supplied_evidence_tokens",
             "latency_end_to_end_seconds", "final_k"]


def verify_seal() -> dict:
    seal = gen.read_json(gen.SEAL_PATH)
    gen.verify_fingerprint(seal, "seal_fingerprint", "prediction seal")
    for name, digest in seal["artifact_sha256"].items():
        if file_sha256(gen.PREDICTIONS_DIR / name) != digest:
            raise SystemExit(f"GATE F: sealed artifact changed after sealing: {name}")
    if file_sha256(gen.PREDICTIONS_DIR / "PREDICTION_MANIFEST.json") != seal["prediction_manifest_file_sha256"]:
        raise SystemExit("GATE F: prediction manifest changed after sealing")
    return seal


# ---------------------------------------------------------------------------
# fetch-gold
# ---------------------------------------------------------------------------

QASPER_TEST_URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz"
MUSIQUE_DRIVE_ID = "1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h"


def download(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    partial = target.with_suffix(target.suffix + ".partial")
    with urllib.request.urlopen(request, timeout=900) as response, partial.open("wb") as stream:
        first = response.read(1 << 20)
        if first.lstrip().startswith(b"<"):
            raise RuntimeError(f"received HTML instead of an archive from {url}")
        stream.write(first)
        while block := response.read(1 << 20):
            stream.write(block)
    partial.replace(target)


def cmd_fetch_gold(args: argparse.Namespace) -> int:
    seal = verify_seal()
    manifest = json.loads((ROOT / "datasets" / "DATASET_MANIFEST.json").read_text(encoding="utf-8"))
    archives = RAW / "_archives"
    archives.mkdir(parents=True, exist_ok=True)
    started = time.time()
    provenance = {"seal_fingerprint": seal["seal_fingerprint"], "sealed_utc": seal["sealed_utc"],
                  "gold_access_started_utc": lq.utc_now(), "files": {}}
    # QASPER test
    qasper_archive = archives / "qasper-test-and-evaluator-v0.3.tgz"
    if not qasper_archive.exists():
        download(QASPER_TEST_URL, qasper_archive)
    expected = manifest["qasper"]["archive_sha256"]["qasper-test-and-evaluator-v0.3.tgz"]
    if file_sha256(qasper_archive) != expected:
        raise SystemExit("QASPER archive SHA-256 mismatch")
    target = RAW / "qasper" / "data" / "qasper-test-v0.3.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(qasper_archive) as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith("qasper-test-v0.3.json"))
        target.write_bytes(tar.extractfile(member).read())
    if file_sha256(target) != manifest["qasper"]["files"]["qasper-test-v0.3.json"]["sha256"]:
        raise SystemExit("qasper-test-v0.3.json SHA-256 mismatch")
    provenance["files"]["qasper_archive"] = {"url": QASPER_TEST_URL, "sha256": expected, "verified": True}
    # MuSiQue-Full
    musique_archive = archives / "musique_v1.0.zip"
    if not musique_archive.exists():
        url = f"https://drive.usercontent.google.com/download?id={MUSIQUE_DRIVE_ID}&export=download&confirm=t"
        download(url, musique_archive)
    expected = manifest["musique_full"]["archive_sha256"]
    if file_sha256(musique_archive) != expected:
        raise SystemExit("MuSiQue archive SHA-256 mismatch")
    target = RAW / "musique" / "data" / "musique_full_v1.0_dev.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(musique_archive) as archive:
        member = next(name for name in archive.namelist() if name.endswith("musique_full_v1.0_dev.jsonl"))
        target.write_bytes(archive.read(member))
    if file_sha256(target) != manifest["musique_full"]["files"]["musique_full_v1.0_dev.jsonl"]["sha256"]:
        raise SystemExit("musique_full_v1.0_dev.jsonl SHA-256 mismatch")
    provenance["files"]["musique_archive"] = {"google_drive_file_id": MUSIQUE_DRIVE_ID, "sha256": expected, "verified": True}
    # Normalize with the unchanged repository adapter and prove file identity with the eligible pool.
    pool_hashes = {row["dataset"]: row["source_file_sha256"] for row in gen.load_jsonl(gen.POOL)}
    for dataset, split in (("qasper", "test"), ("musique_full", "validation")):
        output = NORMALIZED[dataset]
        if not output.exists():
            subprocess.run([sys.executable, "-m", "sar_rag.dataset_adapters", "--dataset", dataset, "--split", split,
                            "--output", str(output)], check=True, cwd=ROOT, env={"PYTHONPATH": str(ROOT / "src")})
        raw, crlf = file_sha256(output), gen.crlf_sha256(output)
        match = "exact" if raw == pool_hashes[dataset] else "crlf_equivalent" if crlf == pool_hashes[dataset] else "MISMATCH"
        if match == "MISMATCH":
            raise SystemExit(f"normalized {dataset} file is not the file the eligible pool was built from")
        provenance["files"][f"normalized_{dataset}"] = {"path": gen.rel(output), "sha256": raw, "crlf_equivalent_sha256": crlf,
                                                        "eligible_pool_recorded_sha256": pool_hashes[dataset], "match": match}
    provenance["gold_access_finished_utc"] = lq.utc_now()
    provenance["seconds"] = time.time() - started
    lq.write_json_atomic(GOLD_PROVENANCE, provenance)
    print(json.dumps(provenance, indent=2))
    return 0


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

def load_gold(ids: set[str]) -> dict[str, dict]:
    gold = {}
    for path in NORMALIZED.values():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["example_id"] in ids:
                    gold[row["example_id"]] = {k: row.get(k) for k in ("gold_answerability", "gold_answers", "gold_evidence_spans")}
    return gold


def nan() -> float:
    return float("nan")


def score_pair(record: dict, label: dict, timing: dict | None) -> dict:
    valid = record["validation_status"] == "valid"
    response = record["validated_response"] if valid else {"answerability": "GENERATION_FAILURE", "claims": []}
    prediction = " ".join(claim["text"] for claim in response["claims"])
    gold_answerability = label.get("gold_answerability")
    if gold_answerability is None:
        f1 = em = answerability_correct = correct_abstention = incorrect_abstention = false_answer = nan()
    else:
        if not valid:
            f1 = em = 0.0
        elif gold_answerability == "INSUFFICIENT":
            f1 = em = float(response["answerability"] == "INSUFFICIENT")
        else:
            f1 = token_f1(prediction, label["gold_answers"])
            em = FROZEN.exact_match(prediction, label["gold_answers"])
        answerability_correct = float(response["answerability"] == gold_answerability)
        correct_abstention = float(gold_answerability == "INSUFFICIENT" and response["answerability"] == "INSUFFICIENT")
        incorrect_abstention = float(gold_answerability != "INSUFFICIENT" and response["answerability"] == "INSUFFICIENT")
        false_answer = float(gold_answerability == "INSUFFICIENT" and response["answerability"] in {"ANSWERABLE", "PARTIALLY_ANSWERABLE"})
    spans = list(dict.fromkeys(label.get("gold_evidence_spans") or []))
    native = FROZEN.covered(record["retrieved_evidence"], spans)
    supplied = FROZEN.covered(record["supplied_evidence"], spans)
    has = bool(spans)
    return {
        "logical_key": record["logical_key"], "dataset": record["dataset"], "question_id": record["question_id"],
        "system": record["system"], "dependency_group": record["source_group_id"],
        "gold_answerability": gold_answerability, "gold_span_count": len(spans),
        "validation_status": record["validation_status"], "valid_generation": float(valid),
        "predicted_answerability": response["answerability"],
        "abstained": float(response["answerability"] == "INSUFFICIENT"),
        "answer_token_f1": f1, "answer_exact_match": em, "answerability_correct": answerability_correct,
        "correct_abstention": correct_abstention, "incorrect_abstention": incorrect_abstention,
        "false_answer_on_insufficient": false_answer,
        "native_evidence_recall": len(native) / len(spans) if has else nan(),
        "supplied_evidence_recall": len(supplied) / len(spans) if has else nan(),
        "answerable_complete_evidence": float(len(supplied) == len(spans)) if has and gold_answerability == "ANSWERABLE" else nan(),
        "insufficient_residual_support_recovery": len(supplied) / len(spans) if has and gold_answerability == "INSUFFICIENT" else nan(),
        "supplied_context_tokens": record["supplied_context_tokens_minilm"],
        "native_context_tokens": record["native_context_tokens_minilm"],
        "qwen_supplied_evidence_tokens": record["qwen_supplied_evidence_tokens"],
        "final_k": record["final_k"], "supplied_k": len(record["supplied_evidence"]),
        "input_tokens": record["input_tokens"], "output_tokens": record["output_tokens"], "total_tokens": record["total_tokens"],
        "latency_end_to_end_seconds": timing["latency_end_to_end_including_save_seconds"] if timing else record["latency_end_to_end_before_save_seconds"],
        "latency_generation_seconds": record["latency_generation_seconds"],
        "output_tokens_per_second": record["throughput_output_tokens_per_second"],
        "peak_allocated_vram_gib": record["peak_allocated_vram_gib"], "peak_reserved_vram_gib": record["peak_reserved_vram_gib"],
        "output_truncated_at_max_new_tokens": float(record["output_truncated_at_max_new_tokens"]),
    }


def mean_defined(values: list[float]) -> float | None:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return statistics.fmean(clean) if clean else None


def cmd_score(args: argparse.Namespace) -> int:
    started = time.time()
    seal = verify_seal()
    gold_provenance = gen.read_json(GOLD_PROVENANCE)
    if gold_provenance["seal_fingerprint"] != seal["seal_fingerprint"]:
        raise SystemExit("gold provenance is not bound to this seal")
    protocol, manifest = gen.load_frozen()
    records = {row["logical_key"]: row for row in gen.load_jsonl(gen.PREDICTIONS_DIR / "PREDICTIONS.jsonl")}
    timings = {row["logical_key"]: row for row in gen.load_jsonl(gen.PREDICTIONS_DIR / "REQUEST_TIMINGS.jsonl")}
    planned = [(q, s) for q in manifest["questions"] for s in lq.SYSTEMS]
    gold = load_gold({q["question_id"] for q in manifest["questions"]})
    rows, missing = [], []
    for question, system in planned:
        key = f"{question['dataset']}|{question['question_id']}|{system}"
        if key not in records:
            missing.append(key)
            continue
        rows.append(score_pair(records[key], gold[question["question_id"]], timings.get(key)))
    per_question = Counter((row["dataset"], row["question_id"]) for row in rows)
    complete = {qid for (_, qid), count in per_question.items() if count == 3}
    paired_rows = [row for row in rows if row["question_id"] in complete]
    METRICS.mkdir(parents=True, exist_ok=True)
    COMPARISONS.mkdir(parents=True, exist_ok=True)
    with (METRICS / "PER_PAIR_METRICS.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    all_metrics = PRIMARY + SECONDARY + ["native_context_tokens", "total_tokens", "latency_generation_seconds",
                                         "output_tokens_per_second", "peak_allocated_vram_gib", "peak_reserved_vram_gib",
                                         "output_truncated_at_max_new_tokens", "supplied_k"]
    aggregates = {}
    for dataset in ("qasper", "musique_full"):
        for system in lq.SYSTEMS:
            subset = [row for row in paired_rows if row["dataset"] == dataset and row["system"] == system]
            aggregates[f"{dataset}|{system}"] = {
                "n_pairs": len(subset),
                "n_defined": {m: sum(1 for r in subset if r[m] is not None and not math.isnan(float(r[m]))) for m in all_metrics},
                "gold_answerability_counts": dict(Counter(str(r["gold_answerability"]) for r in subset)),
                "validation_status_counts": dict(Counter(r["validation_status"] for r in subset)),
                "predicted_answerability_counts": dict(Counter(r["predicted_answerability"] for r in subset)),
                **{m: mean_defined([r[m] for r in subset]) for m in all_metrics},
                "median_latency_end_to_end_seconds": statistics.median([r["latency_end_to_end_seconds"] for r in subset]) if subset else None,
                "final_k_distribution": dict(sorted(Counter(r["final_k"] for r in subset).items())),
            }
    inference = {}
    for dataset in ("qasper", "musique_full"):
        dataset_rows = [row for row in paired_rows if row["dataset"] == dataset]
        for metric in PRIMARY + SECONDARY:
            family = []
            for left, right in COMPARISON_PAIRS:
                name = f"{left}_minus_{right}"
                result = FROZEN.paired_cluster_inference(dataset_rows, metric, left, right)
                result.update(dataset=dataset, metric=metric, comparison=name, role="primary" if metric in PRIMARY else "secondary")
                inference[f"{dataset}|{metric}|{name}"] = result
                if metric in PRIMARY and result["raw_p"] is not None:
                    family.append((name, result["raw_p"]))
            if metric in PRIMARY:
                for name, value in FROZEN.holm(family).items():
                    inference[f"{dataset}|{metric}|{name}"]["holm_adjusted_p"] = value
    sensitivity = {}
    for dataset in ("qasper", "musique_full"):
        valid_questions = {qid for qid in complete if all(
            r["valid_generation"] == 1.0 for r in paired_rows if r["question_id"] == qid)}
        for system in lq.SYSTEMS:
            subset = [r for r in paired_rows if r["dataset"] == dataset and r["system"] == system and r["question_id"] in valid_questions]
            sensitivity[f"{dataset}|{system}"] = {"n": len(subset), **{m: mean_defined([r[m] for r in subset]) for m in PRIMARY + ["answer_exact_match", "answerability_correct"]}}
    results = {
        "status": "SCORED_AFTER_PREDICTION_SEAL_ZERO_JUDGE_CALLS",
        "protocol_fingerprint": protocol["protocol_fingerprint"], "seal_fingerprint": seal["seal_fingerprint"],
        "planned_pairs": len(planned), "scored_pairs": len(rows), "missing_pairs": missing,
        "complete_three_system_questions": len(complete),
        "complete_questions_by_dataset": dict(Counter(ds for (ds, qid) in per_question if qid in complete)),
        "clusters_by_dataset": {ds: len({r["dependency_group"] for r in paired_rows if r["dataset"] == ds}) for ds in ("qasper", "musique_full")},
        "aggregates": aggregates, "paired_inference": inference,
        "sensitivity_all_three_valid_descriptive": sensitivity,
        "statistics": protocol["statistics"], "judge_model_calls": 0, "api_calls": 0,
        "scoring_seconds": time.time() - started, "scored_utc": lq.utc_now(),
    }
    lq.write_json_atomic(METRICS / "RESULTS.json", results)
    fields = ["dataset", "metric", "role", "comparison", "paired_n", "cluster_n", "mean_difference", "ci95_low", "ci95_high", "raw_p", "holm_adjusted_p"]
    with (COMPARISONS / "PAIRED_COMPARISONS.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in inference.values():
            ci = result.get("ci95") or [None, None]
            writer.writerow({"dataset": result["dataset"], "metric": result["metric"], "role": result["role"],
                             "comparison": result["comparison"], "paired_n": result["paired_n"], "cluster_n": result["cluster_n"],
                             "mean_difference": result["mean_difference"], "ci95_low": ci[0], "ci95_high": ci[1],
                             "raw_p": result["raw_p"], "holm_adjusted_p": result.get("holm_adjusted_p")})
    with (METRICS / "DATASET_SYSTEM_RESULTS.csv").open("w", newline="", encoding="utf-8") as stream:
        columns = ["dataset", "system", "n_pairs"] + all_metrics
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for key, value in aggregates.items():
            dataset, system = key.split("|")
            writer.writerow({"dataset": dataset, "system": system, **{c: value.get(c) for c in columns[2:]}})
    print(json.dumps({k: results[k] for k in ("status", "planned_pairs", "scored_pairs", "complete_three_system_questions",
                                              "complete_questions_by_dataset", "clusters_by_dataset", "scoring_seconds")}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch-gold")
    sub.add_parser("score")
    args = parser.parse_args()
    return {"fetch-gold": cmd_fetch_gold, "score": cmd_score}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
