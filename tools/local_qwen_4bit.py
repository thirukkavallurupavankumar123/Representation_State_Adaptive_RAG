"""Reduced local-Qwen (Qwen3-8B, 4-bit NF4) study: inputs, smoke, freeze, generation, sealing.

Gold labels are never read by this tool. Scoring lives in
``tools/score_local_qwen_4bit.py`` and refuses to run before the prediction seal.

Subcommands (run from the repository root with the WSL Qwen environment):

  fetch-inputs   recover the sealed held-out v1.1 LFS request/audit files, SHA-256 verified
  smoke          development smoke on design-exposed smoke-v1.1 pairs
  freeze         derive N from measured smoke throughput; freeze protocol + sample manifest
  generate       reduced held-out generation (one attempt per pair, checkpointed, resumable)
  seal           hash and seal predictions before any gold access
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
import sys
import time
import urllib.request
import uuid


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sar_rag.final_evaluation import ANSWER_SCHEMA, file_sha256, logical_key, object_sha256  # noqa: E402
from sar_rag.generation_v1_1 import build_messages_v1_1  # noqa: E402
from sar_rag.heldout_v1_1 import heldout_request_fingerprint_v1_1  # noqa: E402
from sar_rag import local_qwen_4bit as lq  # noqa: E402


OUT = ROOT / "output" / "local_qwen_4bit"
HELDOUT = ROOT / "output" / "final_evaluation" / "heldout_v1_1"
FINAL_EVAL = ROOT / "output" / "final_evaluation"
INPUTS = OUT / "inputs" / "lfs"
LFS_REMOTE = "https://github.com/thirukkavallurupavankumar123/Representation_State_Adaptive_RAG.git"
LFS_FILES = ("REQUESTS.jsonl", "RETRIEVAL_AND_PACKING_AUDIT.jsonl")

POOL = HELDOUT / "GENERATION_QUESTIONS.jsonl"
HELDOUT_PROTOCOL = HELDOUT / "FINAL_EVALUATION_PROTOCOL_AMENDMENT_V1_1.json"
HELDOUT_MANIFEST = HELDOUT / "FROZEN_QUESTION_SYSTEM_MANIFEST.json"
SEALED_REQUEST_MANIFEST = HELDOUT / "REQUEST_MANIFEST_SEALED.json"
SMOKE_REQUESTS = FINAL_EVAL / "SMOKE_TEST_REQUESTS.jsonl"
SMOKE_AUDIT = FINAL_EVAL / "SMOKE_TEST_RETRIEVAL_AUDIT.jsonl"
V1_MANIFEST = ROOT / "output" / "calibration" / "frozen_manifest.json"
V2_MANIFEST = ROOT / "output" / "adaptive_single_view_v2" / "recalibration_seed20260905" / "FROZEN_V2_MANIFEST.json"

PROTOCOL_PATH = OUT / "protocol" / "PROTOCOL.json"
SAMPLE_MANIFEST_PATH = OUT / "manifests" / "SAMPLE_MANIFEST.json"
FREEZE_RECORD_PATH = OUT / "manifests" / "FREEZE_RECORD.json"
BUDGET_CLOCK_PATH = OUT / "BUDGET_CLOCK.json"
PREDICTIONS_DIR = OUT / "predictions"
SEAL_PATH = OUT / "predictions" / "PREDICTION_SEAL.json"

BUDGET_HOURS = 5.0
SCORING_RESERVE_SECONDS = 30 * 60
EXECUTION_BUFFER_FRACTION = 0.20
CHECKPOINT_OVERHEAD_FRACTION = 0.05
PROVISIONAL_MAX_N = 150
SAMPLING_SEED = lq.MASTER_SEED
GENERATION_CODE = (
    ROOT / "src" / "sar_rag" / "local_qwen_4bit.py",
    ROOT / "src" / "sar_rag" / "generation_v1_1.py",
    ROOT / "src" / "sar_rag" / "final_evaluation.py",
    ROOT / "tools" / "local_qwen_4bit.py",
)
SYSTEM_VERSIONS = {
    "traditional_k10": "sentence_window exact dense index, static top-k=10 (FINAL_EVALUATION_PROTOCOL.json)",
    "sar_rag_v1": "SarRagSystem with output/calibration/frozen_manifest.json (dynamic final k)",
    "adaptive_single_view_v2": "AdaptiveSingleViewV2System with FROZEN_V2_MANIFEST.json (dynamic final k)",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT)).replace("\\", "/")


def write_new_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen artifact: {rel(path)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def body(value: dict, field: str) -> dict:
    return {key: item for key, item in value.items() if key != field}


def with_fingerprint(value: dict, field: str) -> dict:
    value = dict(value)
    value[field] = object_sha256(value)
    return value


def verify_fingerprint(value: dict, field: str, label: str) -> None:
    if object_sha256(body(value, field)) != value[field]:
        raise RuntimeError(f"{label} fingerprint is invalid")


def pointer(path: Path) -> tuple[str, int]:
    fields = dict(line.split(" ", 1) for line in path.read_text(encoding="utf-8").strip().splitlines())
    return fields["oid"].split(":", 1)[1], int(fields["size"])


def utc(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def parse_utc(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def pctl(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] if lower == upper else ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def describe(values: list[float]) -> dict:
    values = [float(v) for v in values if v is not None]
    if not values:
        return {"n": 0}
    return {"n": len(values), "mean": statistics.fmean(values), "median": statistics.median(values),
            "p90": pctl(values, 0.9), "min": min(values), "max": max(values), "sum": sum(values)}


# ---------------------------------------------------------------------------
# fetch-inputs
# ---------------------------------------------------------------------------

def cmd_fetch_inputs(args: argparse.Namespace) -> int:
    sealed = read_json(SEALED_REQUEST_MANIFEST)
    expected = {"REQUESTS.jsonl": sealed["requests_sha256"],
                "RETRIEVAL_AND_PACKING_AUDIT.jsonl": sealed["retrieval_audit_sha256"]}
    INPUTS.mkdir(parents=True, exist_ok=True)
    provenance = {"source": "Git LFS objects referenced by committed pointer files", "remote": LFS_REMOTE,
                  "repository_pointer_files_unchanged": True, "files": {}}
    for name in LFS_FILES:
        oid, size = pointer(HELDOUT / name)
        if oid != expected[name]:
            raise RuntimeError(f"pointer oid for {name} differs from the sealed request manifest")
        target = INPUTS / name
        if not (target.exists() and file_sha256(target) == oid):
            body_ = json.dumps({"operation": "download", "transfers": ["basic"],
                                "objects": [{"oid": oid, "size": size}]}).encode()
            request = urllib.request.Request(LFS_REMOTE + "/info/lfs/objects/batch", data=body_, method="POST", headers={
                "Accept": "application/vnd.git-lfs+json", "Content-Type": "application/vnd.git-lfs+json"})
            with urllib.request.urlopen(request, timeout=60) as response:
                action = json.load(response)["objects"][0]["actions"]["download"]
            partial = target.with_suffix(".partial")
            with urllib.request.urlopen(urllib.request.Request(action["href"], headers=action.get("header", {})), timeout=900) as response, partial.open("wb") as stream:
                while block := response.read(1 << 20):
                    stream.write(block)
            partial.replace(target)
        actual = file_sha256(target)
        if actual != oid or target.stat().st_size != size:
            raise RuntimeError(f"{name}: SHA-256/size mismatch after download")
        provenance["files"][name] = {"pointer_path": rel(HELDOUT / name), "lfs_oid_sha256": oid, "size": size,
                                     "local_cache_path": rel(target), "sha256_verified": actual,
                                     "matches_sealed_request_manifest": True}
    provenance["sealed_request_manifest_fingerprint"] = sealed["request_manifest_fingerprint"]
    provenance["verified_utc"] = lq.utc_now()
    (OUT / "manifests").mkdir(parents=True, exist_ok=True)
    lq.write_json_atomic(OUT / "manifests" / "INPUTS_PROVENANCE.json", provenance)
    print(json.dumps(provenance, indent=2))
    return 0


def crlf_sha256(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    return sha256(data).hexdigest()


def cmd_verify_integrity(args: argparse.Namespace) -> int:
    """Recorded hashes of text artifacts were computed on Windows (CRLF); git stores LF."""
    heldout_protocol = read_json(HELDOUT_PROTOCOL)
    exposure = read_json(HELDOUT / "DATASET_ELIGIBILITY_AND_EXPOSURE.json")
    smoke_protocol = read_json(FINAL_EVAL / "generation_v1_1" / "PROTOCOL.json")
    smoke_validation = read_json(FINAL_EVAL / "generation_v1_1" / "VALIDATION.json")
    manifest = read_json(HELDOUT_MANIFEST)
    sealed = read_json(SEALED_REQUEST_MANIFEST)
    file_checks = {
        rel(POOL): (exposure["generation_questions_sha256"], POOL),
        rel(SMOKE_REQUESTS): (smoke_protocol["source_request_sha256"], SMOKE_REQUESTS),
        rel(FINAL_EVAL / "FINAL_EVALUATION_PROTOCOL.json"): (heldout_protocol["amends_protocol"]["sha256"], FINAL_EVAL / "FINAL_EVALUATION_PROTOCOL.json"),
        rel(FINAL_EVAL / "generation_v1_1/checkpoint/ANSWERS.jsonl"): (smoke_validation["artifacts"]["answers_sha256"], FINAL_EVAL / "generation_v1_1/checkpoint/ANSWERS.jsonl"),
        rel(V1_MANIFEST): (heldout_protocol["systems"]["sar_rag_v1"]["manifest_sha256_reverified"], V1_MANIFEST),
        rel(V2_MANIFEST): (heldout_protocol["systems"]["adaptive_single_view_v2"]["manifest_sha256_reverified"], V2_MANIFEST),
    }
    results = {}
    for name, (expected, path) in file_checks.items():
        raw, crlf = file_sha256(path), crlf_sha256(path)
        results[name] = {"recorded": expected, "worktree_sha256": raw, "crlf_equivalent_sha256": crlf,
                         "verified": expected in {raw, crlf}, "match": "exact" if raw == expected else "crlf_equivalent" if crlf == expected else "MISMATCH"}
    fingerprints = {
        "heldout_v1_1_protocol": object_sha256(body(heldout_protocol, "protocol_fingerprint")) == heldout_protocol["protocol_fingerprint"],
        "heldout_v1_1_question_system_manifest": object_sha256(body(manifest, "manifest_fingerprint")) == manifest["manifest_fingerprint"],
        "heldout_v1_1_sealed_request_manifest": object_sha256(body(sealed, "request_manifest_fingerprint")) == sealed["request_manifest_fingerprint"],
        "smoke_v1_1_protocol": object_sha256(body(smoke_protocol, "protocol_fingerprint")) == smoke_protocol["protocol_fingerprint"],
    }
    report = {"verified_utc": lq.utc_now(), "file_hashes": results, "object_fingerprints_valid": fingerprints,
              "all_verified": all(r["verified"] for r in results.values()) and all(fingerprints.values()),
              "explanation": "Recorded SHA-256 values were computed over CRLF files written on Windows; the git checkout stores LF. A match of the CRLF-equivalent bytes proves identical content. Git-LFS objects are byte-exact and verified separately in INPUTS_PROVENANCE.json."}
    (OUT / "manifests").mkdir(parents=True, exist_ok=True)
    lq.write_json_atomic(OUT / "manifests" / "PRIOR_ARTIFACT_INTEGRITY.json", report)
    print(json.dumps({"all_verified": report["all_verified"], "fingerprints": fingerprints,
                      "files": {k: v["match"] for k, v in results.items()}}, indent=2))
    return 0 if report["all_verified"] else 1


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------

def smoke_selection() -> tuple[list[dict], list[dict], dict]:
    """8 design-exposed smoke-v1.1 questions (24 pairs), chosen by a fixed hash rule."""
    requests = load_jsonl(SMOKE_REQUESTS)
    questions = sorted({(row["dataset"], row["question_id"]) for row in requests})
    strata: dict[str, list[tuple[str, str]]] = {}
    for dataset, question_id in questions:
        stratum = dataset if dataset == "qasper" else f"musique_full:{question_id.rsplit(':', 1)[-1]}"
        strata.setdefault(stratum, []).append((dataset, question_id))
    quota = {"qasper": 4, "musique_full:answerable": 2, "musique_full:insufficient": 2}
    chosen = []
    for stratum, count in sorted(quota.items()):
        ranked = sorted(strata[stratum], key=lambda item: lq.stable_rank(SAMPLING_SEED, "smoke", *item))
        chosen.extend(ranked[:count])
    chosen = sorted(chosen, key=lambda item: lq.stable_rank(SAMPLING_SEED, "smoke-order", *item))
    chosen_set = set(chosen)
    selected = [row for row in requests if (row["dataset"], row["question_id"]) in chosen_set]
    audits = [row for row in load_jsonl(SMOKE_AUDIT) if (row["dataset"], row["question_id"]) in chosen_set]
    rule = {
        "source": rel(SMOKE_REQUESTS), "source_sha256": file_sha256(SMOKE_REQUESTS),
        "population": "16 design-exposed training questions x 3 systems used by successful smoke v1.1",
        "rule": "within strata {qasper:4, musique answerable:2, musique insufficient:2} take lowest "
                "SHA-256('20260930|smoke|dataset|question_id'); order by SHA-256('20260930|smoke-order|...')",
        "questions": [list(item) for item in chosen], "pairs": len(selected),
    }
    return selected, audits, rule


def run_generation(directory: Path, meta: dict, plan: list, requests: dict, pair_extras: dict,
                   model_path: str, deadline_epoch: float | None, fallback_seconds: float,
                   stage: str) -> dict:
    run_id = f"{stage}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    runner = lq.LocalQwenRunner(directory, meta)
    runner.log("model_load_started", run_id=run_id)
    generator = lq.LocalQwenGenerator(model_path)
    warm = generator.warmup()
    env = generator.environment()
    runner.log("model_ready", run_id=run_id, load_seconds=generator.load_seconds, warmup_seconds=warm,
               placement=generator.placement)
    lq.write_json_atomic(directory / f"ENVIRONMENT_{run_id}.json", env)
    summary = runner.run(plan, requests, generator, run_id=run_id, deadline_epoch=deadline_epoch,
                         fallback_seconds_per_pair=fallback_seconds,
                         base_record={"stage": stage}, pair_extras=pair_extras)
    summary.update(run_id=run_id, load_seconds=generator.load_seconds, warmup_seconds=warm)
    return summary


def cmd_smoke(args: argparse.Namespace) -> int:
    if not BUDGET_CLOCK_PATH.exists():
        write_new_json(BUDGET_CLOCK_PATH, {
            "budget_hours": BUDGET_HOURS, "budget_start_utc": lq.utc_now(),
            "definition": "execution-budget clock starts at the first local-Qwen development smoke launch; "
                          "earlier manual setup (download, WSL/env setup, NF4 validation) and this session's "
                          "forensics/implementation are reported separately as setup",
        })
    selected, audits, rule = smoke_selection()
    directory = OUT / "smoke" / args.label
    meta = {"stage": "development_smoke", "label": args.label, "selection": rule,
            "generation_config": lq.generation_config_record(),
            "transport_hash_example": lq.build_qwen_messages("q", [])[1]}
    audit_map = {"|".join((row["dataset"], row["question_id"], row["system"])): row for row in audits}
    plan, requests, extras = [], {}, {}
    unit = 0
    for dataset, question_id in [tuple(item) for item in rule["questions"]]:
        for system in lq.system_order(dataset, question_id, SAMPLING_SEED):
            pair = lq.PlannedPair(dataset, question_id, system, f"smoke:{question_id}", unit)
            plan.append(pair)
            request = next(row for row in selected if logical_key(row) == pair.key)
            requests[pair.key] = request
            audit = audit_map[pair.key]
            extras[pair.key] = {
                "system_version": SYSTEM_VERSIONS[system],
                "final_k": audit["native_final_k"], "packed_final_k": audit["packed_final_k"],
                "supplied_context_tokens_minilm": audit["packed_context_tokens"],
                "stop_reason": audit["stop_reason"], "design_exposed_smoke": True,
            }
        unit += 1
    summary = run_generation(directory, meta, plan, requests, extras, args.model_path, None, 60.0, "smoke")
    report = summarize_run(directory, plan)
    report["run_summary"] = summary
    lq.write_json_atomic(directory / "SMOKE_SUMMARY.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "per_pair"}, indent=2))
    return 0


def summarize_run(directory: Path, plan: list) -> dict:
    predictions = {row["logical_key"]: row for row in load_jsonl(directory / "PREDICTIONS.jsonl")}
    timings = {row["logical_key"]: row for row in load_jsonl(directory / "REQUEST_TIMINGS.jsonl")}
    infra = load_jsonl(directory / "INFRASTRUCTURE_FAILURES.jsonl") if (directory / "INFRASTRUCTURE_FAILURES.jsonl").exists() else []
    rows = [predictions[p.key] for p in plan if p.key in predictions]
    e2e = [timings[r["logical_key"]]["latency_end_to_end_including_save_seconds"] for r in rows if r["logical_key"] in timings]
    return {
        "planned_pairs": len(plan), "completed_pairs": len(rows),
        "validation_status_counts": dict(Counter(r["validation_status"] for r in rows)),
        "parse_status_counts": dict(Counter(r["parse_status"] for r in rows)),
        "answerability_counts_valid": dict(Counter(r["validated_response"]["answerability"] for r in rows if r["validated_response"])),
        "valid_by_system": {s: sum(r["validation_status"] == "valid" for r in rows if r["system"] == s) for s in lq.SYSTEMS},
        "valid_by_dataset": {d: sum(r["validation_status"] == "valid" for r in rows if r["dataset"] == d) for d in ("qasper", "musique_full")},
        "truncation_incidents_max_new_tokens": sum(r["output_truncated_at_max_new_tokens"] for r in rows),
        "thinking_marker_incidents": sum(r["thinking_markers_present"] for r in rows),
        "infrastructure_failures": len(infra),
        "latency_end_to_end_including_save_seconds": describe(e2e),
        "generate_seconds": describe([r["timing"]["generate_seconds"] for r in rows]),
        "prefill_seconds": describe([r["timing"]["prefill_seconds"] for r in rows]),
        "decode_tokens_per_second": describe([r["timing"]["decode_tokens_per_second"] for r in rows]),
        "output_tokens_per_second_generate": describe([r["throughput_output_tokens_per_second"] for r in rows]),
        "input_tokens": describe([r["input_tokens"] for r in rows]),
        "output_tokens": describe([r["output_tokens"] for r in rows]),
        "total_tokens": describe([r["total_tokens"] for r in rows]),
        "qwen_supplied_evidence_tokens": describe([r["qwen_supplied_evidence_tokens"] for r in rows]),
        "qwen_supplied_evidence_tokens_over_1600": sum(r["qwen_supplied_evidence_tokens"] > 1600 for r in rows),
        "supplied_context_tokens_minilm": describe([r.get("supplied_context_tokens_minilm") for r in rows]),
        "max_prompt_plus_allowance_tokens": max((r["prompt_plus_allowance_tokens"] for r in rows), default=None),
        "context_capacity": rows[0]["context_capacity"] if rows else None,
        "final_k_by_system": {s: sorted({r.get("final_k") for r in rows if r["system"] == s}) for s in lq.SYSTEMS},
        "peak_allocated_vram_gib": describe([r["peak_allocated_vram_gib"] for r in rows]),
        "peak_reserved_vram_gib": describe([r["peak_reserved_vram_gib"] for r in rows]),
        "per_pair": [{
            "logical_key": r["logical_key"], "validation_status": r["validation_status"],
            "schema_errors": r["schema_errors"], "semantic_error": r["semantic_error"], "parse_error": r["parse_error"],
            "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"], "finish_reason": r["finish_reason"],
            "e2e_seconds": timings.get(r["logical_key"], {}).get("latency_end_to_end_including_save_seconds"),
        } for r in rows],
    }


# ---------------------------------------------------------------------------
# freeze: runtime estimate, protocol, sample manifest
# ---------------------------------------------------------------------------

def feasible_n(smoke: dict, now_epoch: float, budget_start_epoch: float) -> dict:
    per_pair = smoke["latency_end_to_end_including_save_seconds"]["mean"]
    load_overhead = smoke["run_summary"]["load_seconds"] + smoke["run_summary"]["warmup_seconds"]
    budget_end = budget_start_epoch + BUDGET_HOURS * 3600
    remaining = budget_end - now_epoch
    available = remaining - SCORING_RESERVE_SECONDS - load_overhead
    usable = available * (1 - EXECUTION_BUFFER_FRACTION) * (1 - CHECKPOINT_OVERHEAD_FRACTION)
    per_question = 3 * per_pair
    raw_n = max(0, math.floor(usable / per_question))
    n = min(PROVISIONAL_MAX_N, raw_n)
    n -= n % 4  # equal dataset halves; MuSiQue sampled as 2-question variant groups
    return {
        "budget_start_utc": utc(budget_start_epoch), "budget_end_utc": utc(budget_end),
        "freeze_time_utc": utc(now_epoch), "remaining_budget_seconds": remaining,
        "scoring_reporting_reserve_seconds": SCORING_RESERVE_SECONDS,
        "model_load_and_warmup_seconds": load_overhead,
        "execution_buffer_fraction": EXECUTION_BUFFER_FRACTION,
        "checkpoint_failure_overhead_fraction": CHECKPOINT_OVERHEAD_FRACTION,
        "usable_generation_seconds": usable,
        "smoke_mean_end_to_end_seconds_per_pair": per_pair,
        "smoke_p90_end_to_end_seconds_per_pair": smoke["latency_end_to_end_including_save_seconds"]["p90"],
        "seconds_per_complete_three_system_question": per_question,
        "raw_feasible_questions": raw_n, "provisional_target": PROVISIONAL_MAX_N,
        "frozen_n_questions": n, "frozen_n_answers": 3 * n,
        "generation_hard_deadline_utc": utc(budget_end - SCORING_RESERVE_SECONDS),
        "rule": "N = min(150, floor(0.8 * 0.95 * (budget_end - now - 1800s - load/warmup) / (3 * smoke mean e2e seconds per pair))), rounded down to a multiple of 4",
    }


def build_sample(n: int) -> dict:
    """Equal-probability, label-blind sampling from the audited eligible pool."""
    pool = load_jsonl(POOL)
    qasper = [row for row in pool if row["dataset"] == "qasper"]
    musique_groups: dict[str, list[dict]] = {}
    for row in pool:
        if row["dataset"] == "musique_full":
            musique_groups.setdefault(row["dependency_group"], []).append(row)
    if any(len(rows) != 2 for rows in musique_groups.values()):
        raise RuntimeError("expected exactly two MuSiQue variants per dependency group")
    n_qasper, n_musique = n // 2, n // 2
    picked_qasper = sorted(qasper, key=lambda row: lq.stable_rank(SAMPLING_SEED, "sample", "qasper", row["question_id"]))[:n_qasper]
    picked_groups = sorted(musique_groups, key=lambda group: lq.stable_rank(SAMPLING_SEED, "sample", "musique_full", group))[:n_musique // 2]
    picked = picked_qasper + [row for group in picked_groups for row in musique_groups[group]]
    by_group: dict[str, list[dict]] = {}
    for row in picked:
        by_group.setdefault(row["dependency_group"], []).append(row)
    group_order = sorted(by_group, key=lambda group: lq.stable_rank(SAMPLING_SEED, "group-order", group))
    questions = []
    for unit_index, group in enumerate(group_order):
        for row in sorted(by_group[group], key=lambda item: item["question_id"]):
            questions.append({
                "unit_index": unit_index, "dataset": row["dataset"], "question_id": row["question_id"],
                "dependency_group": row["dependency_group"], "document_key": row["document_key"],
                "query_sha256": row["query_sha256"], "question_seed": lq.question_seed(row["dataset"], row["question_id"]),
                "system_order": lq.system_order(row["dataset"], row["question_id"], SAMPLING_SEED),
            })
    return {
        "questions": questions,
        "counts": {"questions": len(questions), "pairs": 3 * len(questions),
                   "by_dataset": dict(Counter(q["dataset"] for q in questions)),
                   "dependency_groups": len(group_order),
                   "clusters_by_dataset": dict(Counter(by_group[g][0]["dataset"] for g in group_order))},
    }


def cmd_freeze(args: argparse.Namespace) -> int:
    for path in (PROTOCOL_PATH, SAMPLE_MANIFEST_PATH, FREEZE_RECORD_PATH):
        if path.exists():
            raise FileExistsError(f"already frozen: {rel(path)}")
    smoke_dir = OUT / "smoke" / args.accepted_smoke
    smoke = read_json(smoke_dir / "SMOKE_SUMMARY.json")
    if smoke["completed_pairs"] != smoke["planned_pairs"] or smoke["infrastructure_failures"]:
        raise SystemExit("GATE C FAILED: accepted smoke is incomplete or had infrastructure failures")
    if smoke["thinking_marker_incidents"] or smoke["validation_status_counts"].get("valid", 0) < 0.8 * smoke["planned_pairs"]:
        raise SystemExit("GATE C FAILED: systematic thinking or schema problems remain in the accepted smoke")
    clock = read_json(BUDGET_CLOCK_PATH)
    estimate = feasible_n(smoke, time.time(), parse_utc(clock["budget_start_utc"]))
    n = estimate["frozen_n_questions"]
    if n < 40:
        lq.write_json_atomic(OUT / "protocol" / "RUNTIME_ESTIMATE.json", estimate)
        raise SystemExit(f"GATE D FAILED: only {n} questions fit the budget; not launching an underpowered study")
    inputs = read_json(OUT / "manifests" / "INPUTS_PROVENANCE.json")
    heldout_protocol = read_json(HELDOUT_PROTOCOL)
    sealed = read_json(SEALED_REQUEST_MANIFEST)
    env_files = sorted(smoke_dir.glob("ENVIRONMENT_*.json"))
    environment = read_json(env_files[-1])
    _, v1_1_prompt_hash = build_messages_v1_1("q", [])
    protocol = {
        "protocol_version": lq.EXPERIMENT_VERSION,
        "status": "FROZEN_BEFORE_REDUCED_LOCAL_GENERATION",
        "frozen_utc": lq.utc_now(),
        "title": "Reduced local-Qwen paired evaluation of Traditional RAG k=10, SAR-RAG v1 and adaptive single-view v2",
        "scope_statement": "A REDUCED LOCAL STUDY under a five-hour execution-budget design. It is NOT the original full "
                           "held-out evaluation (6,240 questions / 18,720 pairs), which remains preserved and unexecuted.",
        "relationship_to_prior_protocols": {
            "heldout_v1_1_protocol_path": rel(HELDOUT_PROTOCOL),
            "heldout_v1_1_protocol_fingerprint": heldout_protocol["protocol_fingerprint"],
            "heldout_v1_1_modified": False,
            "changed_relative_to_heldout_v1_1": [
                "answer generator: local Qwen/Qwen3-8B 4-bit NF4 instead of Groq openai/gpt-oss-120b",
                "decoding: Qwen non-thinking sampling (T=0.7, top_p=0.8, top_k=20, min_p=0) with per-question seeds instead of T=0",
                "transport: frozen schema rendered as text in the system message (no provider strict-JSON enforcement)",
                "attempts: one attempt per pair; no content-based retries (held-out v1.1 allowed up to three)",
                "sample: reduced label-blind random sample of the same eligible pool",
                "no monetary cost; efficiency reported as local tokens/latency/VRAM",
            ],
            "unchanged": ["eligible exposure-audited pool", "frozen retrieval outputs and packing",
                          "v1.1 user prompt text and system prompt", "answer schema", "semantic validator",
                          "failure scoring", "metrics", "cluster definitions", "bootstrap/randomization design", "Holm families"],
        },
        "model": {"repo": lq.MODEL_REPO, "revision": lq.MODEL_REVISION, "tokenizer_revision": lq.MODEL_REVISION,
                  "local_path": environment["model_path"],
                  "weights_verified": read_json(OUT / "environment" / "qwen3_8b_weight_verification.json")["all_weights_match"],
                  "tokenizer_json_sha256": environment["tokenizer_json_sha256"]},
        "quantization": lq.QUANTIZATION, "compute_dtype": environment["compute_dtype"],
        "placement": environment["placement"],
        "software": {k: environment[k] for k in ("python", "torch", "torch_cuda", "transformers", "accelerate", "bitsandbytes", "tokenizers")},
        "hardware": {k: environment[k] for k in ("gpu", "compute_capability", "vram_total_gib")},
        "generation": lq.GENERATION, "generation_config_record": lq.generation_config_record(),
        "seeds": {"master_seed": lq.MASTER_SEED, "rule": lq.SEED_RULE, "sampling_seed": SAMPLING_SEED,
                  "note": lq.REPEATABILITY_NOTE,
                  "execution_order_independent": True,
                  "same_seed_for_all_three_systems_of_a_question": True},
        "prompt": {"v1_1_prompt_version": "final-evaluation-answer-v1.1", "v1_1_prompt_hash": v1_1_prompt_hash,
                   "transport_version": lq.TRANSPORT_VERSION, "schema": ANSWER_SCHEMA,
                   "schema_transport": "system message = v1.1 system prompt + blank line + fixed instruction + compact schema JSON; user message byte-identical to v1.1",
                   "chat_template": "tokenizer.apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)",
                   "constrained_decoding": False,
                   "strict_schema_enforcement_claimed": False},
        "context_policy": {
            "supplied_evidence_cap": "1,600 tokens counted with the frozen MiniLM retrieval tokenizer by the frozen v1.1 post-retrieval packer (complete chunks only, native order, never truncated)",
            "repacking_for_qwen": False,
            "repacking_rationale": "the frozen cap is defined in retrieval-tokenizer units, independent of the answer model; supplied evidence is therefore byte-identical to the sealed v1.1 requests for all three systems",
            "qwen_token_logging": "Qwen-token counts of supplied evidence and of the full rendered prompt are logged per pair",
            "capacity_check": "rendered prompt tokens + 1,600 output allowance must be <= model max_position_embeddings (40,960); otherwise hard stop, never truncate",
            "silent_truncation": False,
        },
        "retrieval": {
            "systems": SYSTEM_VERSIONS,
            "source": "sealed held-out v1.1 REQUESTS.jsonl + RETRIEVAL_AND_PACKING_AUDIT.jsonl produced by the frozen retrieval code; recovered from Git LFS and SHA-256 verified",
            "requests_sha256": sealed["requests_sha256"], "retrieval_audit_sha256": sealed["retrieval_audit_sha256"],
            "sealed_request_manifest_fingerprint": sealed["request_manifest_fingerprint"],
            "per_request_integrity": "every sampled request must reproduce its sealed heldout_request_fingerprint_v1_1",
            "frozen_manifests_sha256": {"sar_rag_v1": file_sha256(V1_MANIFEST), "adaptive_single_view_v2": file_sha256(V2_MANIFEST)},
            "dynamic_final_k_preserved": True, "recalibration": False, "inputs_provenance": inputs,
        },
        "attempt_and_failure_policy": {
            "attempts_per_pair": 1, "content_based_retries": False,
            "invalid_generation": "parse/schema/semantic failures are retained in the denominator and scored with the frozen persistent-failure rule: answer F1 = EM = 0, answerability = GENERATION_FAILURE (incorrect, never abstention); retrieval metrics scored from the sealed logs",
            "repair": "prohibited: no field insertion, type coercion, enum rewriting, evidence-ID insertion or JSON correction",
            "infrastructure_failures": "exceptions (CUDA/OOM/process kill/disk) are logged in INFRASTRUCTURE_FAILURES.jsonl, never recorded as model outputs; the run stops and the identical command resumes with unchanged seeds; completed pairs are skipped only when their fingerprint matches",
            "time_expiry": "the runner stops before starting a dependency-group unit that cannot finish before the hard deadline; the planned manifest is not redefined; missing pairs are reported as missing",
        },
        "sampling": {
            "eligible_pool": rel(POOL), "eligible_pool_sha256": file_sha256(POOL),
            "pool_counts": {"qasper": 1408, "musique_full": 4832},
            "n_questions": n, "n_pairs": 3 * n, "split": {"qasper": n // 2, "musique_full": n // 2},
            "qasper": "equal-probability sample of questions without replacement: lowest SHA-256('20260930|sample|qasper|question_id')",
            "musique_full": "equal-probability sample of dependency groups (both answerable and insufficient variants kept together): lowest SHA-256('20260930|sample|musique_full|dependency_group')",
            "metadata_used": ["dataset", "question_id", "dependency_group"],
            "labels_used": False,
            "stratification": "by dataset only (fixed 50/50); no stratification on answerability, evidence or outcomes",
            "execution_order": "dependency-group units ordered by SHA-256('20260930|group-order|dependency_group'); questions by question_id within a unit; system order per question by SHA-256('20260930|system-order|dataset|question_id|system')",
        },
        "runtime_estimate": estimate,
        "development_smoke": {
            "accepted_run": rel(smoke_dir), "summary_sha256": file_sha256(smoke_dir / "SMOKE_SUMMARY.json"),
            "pairs": smoke["planned_pairs"], "validation_status_counts": smoke["validation_status_counts"],
            "answerability_counts_valid": smoke["answerability_counts_valid"],
            "configuration_changes_after_smoke": [],
            "observations": [
                "invalid smoke outputs were uncited 'missing information' claims rejected by the frozen validator (model behavior, not transport)",
                "Qwen labelled no valid smoke output ANSWERABLE (mostly PARTIALLY_ANSWERABLE); scientific instructions deliberately NOT changed",
                "no parse, schema-type, thinking-marker, truncation, offload or infrastructure problems",
            ],
            "repeatability": {path.name: file_sha256(path) for path in sorted((OUT / "smoke").glob("REPEATABILITY_*.json"))},
        },
        "scoring": {
            "gold_access": "only after PREDICTION_SEAL.json exists and verifies; gold normalized files must reproduce the SHA-256 recorded in the eligible pool",
            "implementation": "tools/score_local_qwen_4bit.py reusing token_f1 (final_evaluation.py) and normalize/covered/exact_match/paired_cluster_inference/holm (tools/score_final_heldout_v1_1.py) unchanged",
            "answer": ["answer token F1", "Exact Match"],
            "evidence": ["supplied-evidence unique gold-span recall (primary)", "native retrieved-evidence recall (secondary, labeled as retrieved)",
                         "answerable-only complete-evidence recovery (supplied)", "insufficient-question residual-support recovery (supplied)"],
            "behavior": ["answerability accuracy", "correct abstention", "incorrect abstention", "false answer on insufficient",
                         "abstention rate", "parse validity", "schema validity", "semantic validity", "invalid generation rate"],
            "efficiency": ["rendered prompt tokens", "output tokens", "Qwen supplied-evidence tokens", "MiniLM supplied-context tokens",
                           "end-to-end latency", "generate latency", "output tokens/s", "peak VRAM"],
            "prediction_text": "space-joined claim texts (heldout v1.1 scorer rule); INSUFFICIENT gold scores F1 = EM = 1 only for a valid INSUFFICIENT prediction",
            "undefined_gold_answerability": "normalized QASPER records without gold_answerability (no usable reference answer and not unanimously unanswerable) get NaN answer/answerability/abstention metrics and are counted; evidence metrics still apply when gold spans exist (the unexecuted heldout v1.1 scorer would raise KeyError on such rows; no other scorer rule changes)",
            "missing_pairs": "pairs never attempted (time expiry or unresolved infrastructure stop) are reported as missing, not scored as model failures; paired comparisons use questions with all three systems attempted; attempted-but-invalid generations stay in the denominator as failures",
            "gold_file_identity": "regenerated normalized gold files must match eligible-pool source_file_sha256 exactly or as CRLF-equivalent bytes",
        },
        "statistics": {
            "comparisons": ["adaptive_single_view_v2 - sar_rag_v1", "sar_rag_v1 - traditional_k10", "adaptive_single_view_v2 - traditional_k10"],
            "primary_metrics": ["answer_token_f1", "supplied_evidence_recall", "answerable_complete_evidence", "supplied_context_tokens"],
            "confidence_intervals": "10,000-replicate paired cluster bootstrap, 95% percentile, seed 20260926 (heldout v1.1 implementation)",
            "tests": "paired cluster sign-randomization, 10,000 draws (heldout v1.1 implementation)",
            "clusters": {"qasper": "paper (qasper-paper:*)", "musique_full": "source group (musique-group:*)"},
            "multiplicity": "Holm across the three comparisons within each dataset x primary-metric family",
            "secondary_metrics": "Exact Match, answerability/abstention, validity and efficiency metrics get the same CI/test machinery but no multiplicity claim; reported as secondary",
            "datasets_reported_separately": True, "no_cross_dataset_average": True,
            "interpretation": "p > 0.05 or a CI crossing zero is reported as 'not statistically distinguishable in this study', never as equivalence; no equivalence test is declared",
            "sensitivity": "descriptive only: metrics restricted to questions where all three systems produced valid generations",
        },
        "interpretation_limits": [
            "reduced sample under a five-hour budget; not the full held-out evaluation",
            "single generator family (Qwen3-8B), 8B capacity, 4-bit NF4 quantization",
            "stochastic decoding; one sample per pair",
            "hardware/software specific; fixed seeds do not guarantee cross-platform identity",
            "invalid generations counted as failures in the denominator",
            "valid evidence IDs do not establish claim faithfulness",
            "MuSiQue is development-split evidence; QASPER test supports dataset-specific test claims only",
            "public-benchmark pretraining exposure of Qwen3 is unknown",
        ],
        "code_sha256_bound_at_freeze": {rel(path): file_sha256(path) for path in GENERATION_CODE},
    }
    protocol = with_fingerprint(protocol, "protocol_fingerprint")
    write_new_json(PROTOCOL_PATH, protocol)
    sample = build_sample(n)
    sealed_pairs = {row["logical_key"]: row for row in sealed["pairs"]}
    for question in sample["questions"]:
        question["sealed_request_fingerprints_v1_1"] = {
            system: sealed_pairs[f"{question['dataset']}|{question['question_id']}|{system}"]["heldout_request_fingerprint_v1_1"]
            for system in lq.SYSTEMS}
    manifest = with_fingerprint({
        "manifest_version": f"{lq.EXPERIMENT_VERSION}-sample-1",
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "eligible_pool": rel(POOL), "eligible_pool_sha256": file_sha256(POOL),
        "sampling_seed": SAMPLING_SEED, "systems": list(lq.SYSTEMS),
        "inclusion_rules": protocol["sampling"], "exclusion_rules": "none beyond the frozen heldout_v1_1 exposure audit",
        **sample,
    }, "manifest_fingerprint")
    write_new_json(SAMPLE_MANIFEST_PATH, manifest)
    record = {
        "protocol_path": rel(PROTOCOL_PATH), "protocol_file_sha256": file_sha256(PROTOCOL_PATH),
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "sample_manifest_path": rel(SAMPLE_MANIFEST_PATH), "sample_manifest_file_sha256": file_sha256(SAMPLE_MANIFEST_PATH),
        "sample_manifest_fingerprint": manifest["manifest_fingerprint"],
        "accepted_smoke": rel(smoke_dir), "accepted_smoke_summary_sha256": file_sha256(smoke_dir / "SMOKE_SUMMARY.json"),
        "frozen_utc": lq.utc_now(), "gold_labels_accessed": False,
    }
    write_new_json(FREEZE_RECORD_PATH, record)
    (OUT / "protocol" / "PROTOCOL.sha256").write_text(f"{record['protocol_file_sha256']}  PROTOCOL.json\n", encoding="utf-8")
    (OUT / "manifests" / "SAMPLE_MANIFEST.sha256").write_text(f"{record['sample_manifest_file_sha256']}  SAMPLE_MANIFEST.json\n", encoding="utf-8")
    print(json.dumps({"n_questions": n, **record, "estimate": estimate, "counts": sample["counts"]}, indent=2))
    return 0


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

def load_frozen() -> tuple[dict, dict]:
    record = read_json(FREEZE_RECORD_PATH)
    if file_sha256(PROTOCOL_PATH) != record["protocol_file_sha256"] or file_sha256(SAMPLE_MANIFEST_PATH) != record["sample_manifest_file_sha256"]:
        raise RuntimeError("GATE E: protocol or sample manifest file changed after freeze")
    protocol, manifest = read_json(PROTOCOL_PATH), read_json(SAMPLE_MANIFEST_PATH)
    verify_fingerprint(protocol, "protocol_fingerprint", "protocol")
    verify_fingerprint(manifest, "manifest_fingerprint", "sample manifest")
    if manifest["protocol_fingerprint"] != protocol["protocol_fingerprint"]:
        raise RuntimeError("sample manifest is not linked to the frozen protocol")
    return protocol, manifest


def load_sampled_inputs(protocol: dict, manifest: dict) -> tuple[dict, dict]:
    wanted = {f"{q['dataset']}|{q['question_id']}|{s}": q for q in manifest["questions"] for s in lq.SYSTEMS}
    for name, expected in (("REQUESTS.jsonl", protocol["retrieval"]["requests_sha256"]),
                           ("RETRIEVAL_AND_PACKING_AUDIT.jsonl", protocol["retrieval"]["retrieval_audit_sha256"])):
        if file_sha256(INPUTS / name) != expected:
            raise RuntimeError(f"sealed input {name} does not match its frozen SHA-256")
    requests, audits = {}, {}
    with (INPUTS / "REQUESTS.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = logical_key(row)
            if key in wanted:
                expected = wanted[key]["sealed_request_fingerprints_v1_1"][row["system"]]
                if heldout_request_fingerprint_v1_1(row) != expected:
                    raise RuntimeError(f"sealed request integrity failure: {key}")
                requests[key] = row
    with (INPUTS / "RETRIEVAL_AND_PACKING_AUDIT.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["logical_key"] in wanted:
                audits[row["logical_key"]] = row
    if set(requests) != set(wanted) or set(audits) != set(wanted):
        raise RuntimeError("sampled keys missing from sealed inputs")
    for key, request in requests.items():
        supplied = [{"evidence_id": item["evidence_id"], "text": item["text"]} for item in audits[key]["supplied_evidence"]]
        if supplied != request["evidence"]:
            raise RuntimeError(f"request evidence differs from audited supplied evidence: {key}")
        if wanted[key]["query_sha256"] != sha256(request["question"].encode("utf-8")).hexdigest():
            raise RuntimeError(f"question text differs from the eligible pool: {key}")
    return requests, audits


def audit_extras(audit: dict) -> dict:
    return {
        "system_version": SYSTEM_VERSIONS[audit["system"]],
        "retrieval_config_hash": audit["configuration_fingerprint"],
        "retrieved_evidence": audit["native_evidence"],
        "retrieved_evidence_ids": [item["chunk_id"] for item in audit["native_evidence"]],
        "retrieved_order": [item["native_rank"] for item in audit["native_evidence"]],
        "retrieved_k": len(audit["native_evidence"]),
        "final_k": audit["native_final_k"],
        "native_context_tokens_minilm": audit["native_context_tokens"],
        "supplied_evidence_audit": [{k: v for k, v in item.items() if k != "text"} for item in audit["supplied_evidence"]],
        "supplied_evidence_chunk_ids": [item["chunk_id"] for item in audit["supplied_evidence"]],
        "supplied_evidence_native_ranks": [item["native_rank"] for item in audit["supplied_evidence"]],
        "supplied_context_tokens_minilm": audit["supplied_context_tokens"],
        "packing_exclusions": audit["packing_exclusions"],
        "packing_truncated_chunk_count": audit["packing_truncated_chunk_count"],
        "stop_reason": audit["stop_reason"],
    }


def cmd_generate(args: argparse.Namespace) -> int:
    if SEAL_PATH.exists():
        raise SystemExit("predictions are already sealed; generation is closed")
    protocol, manifest = load_frozen()
    drift = {path: sha for path, sha in protocol["code_sha256_bound_at_freeze"].items() if file_sha256(ROOT / path) != sha}
    if drift:
        raise SystemExit(f"generation code changed after freeze: {sorted(drift)}")
    requests, audits = load_sampled_inputs(protocol, manifest)
    plan = [lq.PlannedPair(q["dataset"], q["question_id"], s, q["dependency_group"], q["unit_index"])
            for q in manifest["questions"] for s in q["system_order"]]
    meta = {"stage": "reduced_evaluation", "protocol_fingerprint": protocol["protocol_fingerprint"],
            "sample_manifest_fingerprint": manifest["manifest_fingerprint"],
            "generation_config": lq.generation_config_record(), "planned_pairs": len(plan)}
    deadline = parse_utc(protocol["runtime_estimate"]["generation_hard_deadline_utc"])
    fallback = protocol["runtime_estimate"]["smoke_mean_end_to_end_seconds_per_pair"]
    extras = {key: audit_extras(audit) for key, audit in audits.items()}
    summary = run_generation(PREDICTIONS_DIR, meta, plan, requests, extras, args.model_path, deadline, fallback, "reduced_eval")
    print(json.dumps(summary, indent=2))
    return 0


# ---------------------------------------------------------------------------
# seal
# ---------------------------------------------------------------------------

def cmd_seal(args: argparse.Namespace) -> int:
    if SEAL_PATH.exists():
        raise FileExistsError("prediction seal already exists")
    protocol, manifest = load_frozen()
    rows, torn = lq.load_jsonl_tolerant(PREDICTIONS_DIR / "PREDICTIONS.jsonl")
    log = load_jsonl(PREDICTIONS_DIR / "RUN_LOG.jsonl")
    starts = [e["run_id"] for e in log if e["event"] == "run_started"]
    ends = {e["run_id"] for e in log if e["event"] in {"run_finished", "run_aborted_infrastructure"}}
    planned = {f"{q['dataset']}|{q['question_id']}|{s}": q for q in manifest["questions"] for s in lq.SYSTEMS}
    completed = {row["logical_key"]: row for row in rows}
    if not set(completed) <= set(planned):
        raise RuntimeError("predictions contain unplanned pairs")
    by_question = Counter((row["dataset"], row["question_id"]) for row in rows)
    complete_questions = sorted(q for q, count in by_question.items() if count == 3)
    missing = sorted(set(planned) - set(completed))
    files = ["PREDICTIONS.jsonl", "VALIDATION_ERRORS.jsonl", "INFRASTRUCTURE_FAILURES.jsonl",
             "REQUEST_TIMINGS.jsonl", "RUN_LOG.jsonl", "CHECKPOINT_META.json"]
    hashes = {name: file_sha256(PREDICTIONS_DIR / name) for name in files if (PREDICTIONS_DIR / name).exists()}
    hashes.update({path.name: file_sha256(path) for path in sorted(PREDICTIONS_DIR.glob("ENVIRONMENT_*.json"))})
    prediction_manifest = {
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "sample_manifest_fingerprint": manifest["manifest_fingerprint"],
        "pairs": [{"logical_key": row["logical_key"], "request_fingerprint": row["request_fingerprint"],
                   "validation_status": row["validation_status"], "raw_output_sha256": object_sha256(row["raw_output"]),
                   "record_sha256": object_sha256(row)} for row in sorted(rows, key=lambda r: r["logical_key"])],
    }
    prediction_manifest = with_fingerprint(prediction_manifest, "prediction_manifest_fingerprint")
    manifest_path = PREDICTIONS_DIR / "PREDICTION_MANIFEST.json"
    write_new_json(manifest_path, prediction_manifest)
    seal = {
        "status": "SEALED_BEFORE_GOLD_ACCESS",
        "sealed_utc": lq.utc_now(),
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "sample_manifest_fingerprint": manifest["manifest_fingerprint"],
        "prediction_manifest_file_sha256": file_sha256(manifest_path),
        "prediction_manifest_fingerprint": prediction_manifest["prediction_manifest_fingerprint"],
        "artifact_sha256": hashes,
        "planned_questions": len(manifest["questions"]), "planned_pairs": len(planned),
        "completed_pairs": len(completed), "complete_three_system_questions": len(complete_questions),
        "questions_attempted": len(by_question), "missing_pairs": missing,
        "validation_status_counts": dict(Counter(row["validation_status"] for row in rows)),
        "infrastructure_failure_records": len(load_jsonl(PREDICTIONS_DIR / "INFRASTRUCTURE_FAILURES.jsonl")) if (PREDICTIONS_DIR / "INFRASTRUCTURE_FAILURES.jsonl").exists() else 0,
        "torn_checkpoint_lines": torn, "runs_started": starts, "runs_without_clean_end": sorted(set(starts) - ends),
        "gold_labels_accessed_before_seal": False,
    }
    seal = with_fingerprint(seal, "seal_fingerprint")
    write_new_json(SEAL_PATH, seal)
    for path in list(PREDICTIONS_DIR.iterdir()):
        if path.is_file():
            try:
                path.chmod(0o444)  # best effort on drvfs; hashes in the seal are the real guard
            except OSError:
                pass
    print(json.dumps({k: v for k, v in seal.items() if k != "missing_pairs"} | {"missing_pair_count": len(missing)}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch-inputs")
    sub.add_parser("verify-integrity")
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--label", required=True)
    smoke.add_argument("--model-path", default=lq.DEFAULT_MODEL_PATH)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--accepted-smoke", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--model-path", default=lq.DEFAULT_MODEL_PATH)
    sub.add_parser("seal")
    args = parser.parse_args()
    return {"fetch-inputs": cmd_fetch_inputs, "verify-integrity": cmd_verify_integrity, "smoke": cmd_smoke, "freeze": cmd_freeze,
            "generate": cmd_generate, "seal": cmd_seal}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
