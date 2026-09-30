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
    if not gen.SEAL_PATH.exists():
        raise SystemExit("GATE F: predictions are not sealed; gold labels may not be accessed")
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


# ---------------------------------------------------------------------------
# report tables + timeline
# ---------------------------------------------------------------------------

def fmt(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def run_log_span(directory: Path) -> dict:
    events = gen.load_jsonl(directory / "RUN_LOG.jsonl")
    first = min(e["utc"] for e in events)
    last = max(e["utc"] for e in events)
    loads = [e for e in events if e["event"] == "model_ready"]
    return {"start_utc": first, "end_utc": last, "seconds": gen.parse_utc(last) - gen.parse_utc(first),
            "model_load_seconds": [e["load_seconds"] for e in loads], "warmup_seconds": [e["warmup_seconds"] for e in loads],
            "runs": [e["run_id"] for e in events if e["event"] == "run_started"],
            "stop_reasons": [e.get("stop_reason") for e in events if e["event"] == "run_finished"]}


def cmd_report(args: argparse.Namespace) -> int:
    seal = verify_seal()
    results = gen.read_json(METRICS / "RESULTS.json")
    protocol, manifest = gen.load_frozen()
    gold = gen.read_json(GOLD_PROVENANCE)
    clock = gen.read_json(gen.BUDGET_CLOCK_PATH)
    budget_start = gen.parse_utc(clock["budget_start_utc"])
    smoke = {label: run_log_span(OUT / "smoke" / label) for label in ("r1", "r1_repeat")}
    generation = run_log_span(gen.PREDICTIONS_DIR)
    now = time.time()
    timeline = {
        "manual_setup_before_session": {
            "first_recorded_artifact_utc": "2026-09-30T10:12:45Z (environment/gpu_report.txt)",
            "qwen_weights_downloaded_by_utc": "2026-09-30T10:20:42Z (qwen3_8b_weight_hashes.csv)",
            "wsl_environment_created_utc": "2026-09-30T12:14:40Z (sar_rag_qwen_env/pyvenv.cfg)",
            "nf4_validation_completed_utc": "2026-09-30T12:17:43Z (qwen_wsl_gpu_validation.json)",
            "recoverable_span_seconds": 2 * 3600 + 4 * 60 + 58,
            "caveat": "lower bound of the recorded-artifact span only; work before 10:12:45Z and idle time are not recoverable",
        },
        "session_engineering_before_budget_clock": {
            "first_recorded_session_timestamp_utc": "2026-09-30T14:09:06Z",
            "budget_clock_start_utc": clock["budget_start_utc"],
            "note": "repository forensics, weight-hash verification, LFS input recovery and implementation; reported as setup",
        },
        "smoke": smoke,
        "freeze_utc": protocol["frozen_utc"],
        "generation": generation,
        "seal_utc": seal["sealed_utc"],
        "gold_fetch": {"start_utc": gold["gold_access_started_utc"], "end_utc": gold["gold_access_finished_utc"], "seconds": gold["seconds"]},
        "scoring_seconds": results["scoring_seconds"], "scored_utc": results["scored_utc"],
        "report_tables_utc": lq.utc_now(),
        "budget_elapsed_seconds_at_report_tables": now - budget_start,
        "budget_seconds": gen.BUDGET_HOURS * 3600,
    }
    records = gen.load_jsonl(gen.PREDICTIONS_DIR / "PREDICTIONS.jsonl")
    per_pair = gen.load_jsonl(METRICS / "PER_PAIR_METRICS.jsonl")
    efficiency = {
        "pairs": len(records),
        "input_tokens": gen.describe([r["input_tokens"] for r in records]),
        "output_tokens": gen.describe([r["output_tokens"] for r in records]),
        "total_tokens": gen.describe([r["total_tokens"] for r in records]),
        "qwen_supplied_evidence_tokens": gen.describe([r["qwen_supplied_evidence_tokens"] for r in records]),
        "qwen_supplied_evidence_tokens_over_1600": sum(r["qwen_supplied_evidence_tokens"] > 1600 for r in records),
        "supplied_context_tokens_minilm": gen.describe([r["supplied_context_tokens_minilm"] for r in records]),
        "latency_end_to_end_seconds": gen.describe([p["latency_end_to_end_seconds"] for p in per_pair]),
        "latency_generation_seconds": gen.describe([r["latency_generation_seconds"] for r in records]),
        "prefill_seconds": gen.describe([r["timing"]["prefill_seconds"] for r in records]),
        "output_tokens_per_second": gen.describe([r["throughput_output_tokens_per_second"] for r in records]),
        "decode_tokens_per_second": gen.describe([r["timing"]["decode_tokens_per_second"] for r in records]),
        "peak_allocated_vram_gib": gen.describe([r["peak_allocated_vram_gib"] for r in records]),
        "peak_reserved_vram_gib": gen.describe([r["peak_reserved_vram_gib"] for r in records]),
        "max_prompt_plus_allowance_tokens": max(r["prompt_plus_allowance_tokens"] for r in records),
        "truncations_at_max_new_tokens": sum(r["output_truncated_at_max_new_tokens"] for r in records),
        "thinking_marker_incidents": sum(r["thinking_markers_present"] for r in records),
        "validation_status_counts": dict(Counter(r["validation_status"] for r in records)),
        "validation_status_by_system": {s: dict(Counter(r["validation_status"] for r in records if r["system"] == s)) for s in lq.SYSTEMS},
        "invalid_error_kinds": dict(Counter((r["semantic_error"] or (r["schema_errors"] or [""])[0] or r["parse_error"] or "")[:80] for r in records if r["validation_status"] != "valid")),
    }
    lines = ["# Result tables (generated)", "",
             f"Protocol `{protocol['protocol_fingerprint']}`; seal `{seal['seal_fingerprint']}`.", ""]
    headers = ["system", "n", "valid", "F1", "EM", "ans-acc", "abstain", "supplied recall", "retrieved recall",
               "complete ev. (ans)", "residual (insuf)", "final k", "supplied tok (MiniLM)", "Qwen ev. tok", "input tok", "output tok", "e2e s"]
    for dataset in ("qasper", "musique_full"):
        lines += [f"## {dataset}: per-system estimates (complete three-system questions)", "",
                  "| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
        for system in lq.SYSTEMS:
            a = results["aggregates"][f"{dataset}|{system}"]
            lines.append("| " + " | ".join([system, str(a["n_pairs"]), fmt(a["valid_generation"]), fmt(a["answer_token_f1"]),
                                            fmt(a["answer_exact_match"]), fmt(a["answerability_correct"]), fmt(a["abstained"]),
                                            fmt(a["supplied_evidence_recall"]), fmt(a["native_evidence_recall"]),
                                            fmt(a["answerable_complete_evidence"]), fmt(a["insufficient_residual_support_recovery"]),
                                            fmt(a["final_k"], 1), fmt(a["supplied_context_tokens"], 0), fmt(a["qwen_supplied_evidence_tokens"], 0),
                                            fmt(a["input_tokens"], 0), fmt(a["output_tokens"], 0), fmt(a["latency_end_to_end_seconds"], 2)]) + " |")
        a0 = results["aggregates"][f"{dataset}|traditional_k10"]
        lines += ["", f"Gold answerability counts per system: {a0['gold_answerability_counts']}; defined-metric counts (traditional): "
                      f"F1 {a0['n_defined']['answer_token_f1']}, supplied recall {a0['n_defined']['supplied_evidence_recall']}, "
                      f"complete evidence {a0['n_defined']['answerable_complete_evidence']}, residual {a0['n_defined']['insufficient_residual_support_recovery']}.", ""]
        for role in ("primary", "secondary"):
            lines += [f"### {dataset}: paired differences ({role})", "",
                      "| metric | comparison | paired n | clusters | mean diff | 95% CI | raw p | Holm p |", "|---|---|---|---|---|---|---|---|"]
            for result in results["paired_inference"].values():
                if result["dataset"] == dataset and result["role"] == role:
                    ci = result["ci95"] or [None, None]
                    lines.append(f"| {result['metric']} | {result['comparison']} | {result['paired_n']} | {result['cluster_n']} | "
                                 f"{fmt(result['mean_difference'], 4)} | [{fmt(ci[0], 4)}, {fmt(ci[1], 4)}] | {fmt(result['raw_p'], 4)} | "
                                 f"{fmt(result.get('holm_adjusted_p'), 4)} |")
            lines.append("")
        lines += [f"### {dataset}: sensitivity (descriptive; questions where all three generations were valid)", "",
                  "| system | n | F1 | EM | ans-acc | supplied recall |", "|---|---|---|---|---|---|"]
        for system in lq.SYSTEMS:
            s = results["sensitivity_all_three_valid_descriptive"][f"{dataset}|{system}"]
            lines.append(f"| {system} | {s['n']} | {fmt(s['answer_token_f1'])} | {fmt(s['answer_exact_match'])} | {fmt(s['answerability_correct'])} | {fmt(s['supplied_evidence_recall'])} |")
        lines.append("")
    (OUT / "reports").mkdir(parents=True, exist_ok=True)
    (OUT / "reports" / "RESULT_TABLES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    lq.write_json_atomic(OUT / "reports" / "TIMELINE.json", timeline)
    lq.write_json_atomic(OUT / "reports" / "EFFICIENCY_SUMMARY.json", efficiency)
    print(json.dumps({"timeline": timeline, "efficiency": efficiency}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch-gold")
    sub.add_parser("score")
    sub.add_parser("report")
    args = parser.parse_args()
    return {"fetch-gold": cmd_fetch_gold, "score": cmd_score, "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
