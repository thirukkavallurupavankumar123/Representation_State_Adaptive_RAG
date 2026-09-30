"""Freeze the held-out v1.1 evaluation package without reading gold labels.

This preparation step deliberately projects the normalized datasets onto only
the fields needed for generation (question, documents, IDs, and dependency
groups).  Gold answers, answerability labels, and evidence labels are neither
read nor copied.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sar_rag.final_evaluation import file_sha256, object_sha256  # noqa: E402
from sar_rag.generation_v1_1 import (  # noqa: E402
    MAX_COMPLETION_TOKENS,
    MODEL,
    REASONING_EFFORT,
    SCHEMA_NAME,
    STRICT,
    TEMPERATURE,
    TOP_P,
    USER_TEMPLATE_VERSION,
    generation_config_v1_1,
)


OUT = ROOT / "output" / "final_evaluation" / "heldout_v1_1"
SYSTEMS = ("traditional_k10", "sar_rag_v1", "adaptive_single_view_v2")
QASPER = ROOT / "datasets" / "normalized" / "qasper_test.jsonl"
MUSIQUE = ROOT / "datasets" / "normalized" / "musique_full_validation.jsonl"
V1 = ROOT / "output" / "calibration" / "frozen_manifest.json"
V2 = ROOT / "output" / "adaptive_single_view_v2" / "recalibration_seed20260905" / "FROZEN_V2_MANIFEST.json"
EXPOSED_TRAIN = ROOT / "output" / "adaptive_single_view_v2" / "DESIGN_EXPOSED_TRAIN_IDS.json"
EXPOSED_ALL = ROOT / "output" / "adaptive_single_view_v2" / "DESIGN_EXPOSED_IDS.json"
HELDOUT_REGISTRY = ROOT / "output" / "adaptive_single_view_v2" / "HELDOUT_ID_REGISTRY.json"
SMOKE_VALIDATION = ROOT / "output" / "final_evaluation" / "generation_v1_1" / "VALIDATION.json"
SMOKE_PROTOCOL = ROOT / "output" / "final_evaluation" / "generation_v1_1" / "PROTOCOL.json"
SMOKE_MANIFEST = ROOT / "output" / "final_evaluation" / "generation_v1_1" / "RUN_MANIFEST.json"
SMOKE_ANSWERS = ROOT / "output" / "final_evaluation" / "generation_v1_1" / "checkpoint" / "ANSWERS.jsonl"
SMOKE_ATTEMPTS = ROOT / "output" / "final_evaluation" / "generation_v1_1" / "checkpoint" / "API_ATTEMPTS.jsonl"
ORIGINAL_PROTOCOL = ROOT / "output" / "final_evaluation" / "FINAL_EVALUATION_PROTOCOL.json"

PRICING = {
    "input_per_million": 0.15,
    "cached_input_per_million": 0.075,
    "output_per_million": 0.60,
    "currency": "USD",
    "source": "https://console.groq.com/docs/model/openai/gpt-oss-120b",
    "verified_date": "2026-09-26",
}
MAX_PER_PAIR = 3
MINIMUM_INTERVAL_SECONDS = 30
CONSERVATIVE_INPUT_TOKENS = 2800
SMOKE_INPUT_TOKENS = 75897
SMOKE_OUTPUT_TOKENS = 24168
SMOKE_PAIRS = 48
PRIOR_OUTCOME_ARTIFACTS = (
    ROOT / "output/experiments/pilot_combined/paired_metrics.csv",
    ROOT / "output/experiments/pilot_combined/results.json",
    ROOT / "output/experiments/pilot_musique_1/checkpoint.json",
    ROOT / "output/experiments/pilot_musique_1/paired_metrics.csv",
    ROOT / "output/experiments/pilot_musique_1/results.json",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_new(path: Path, value: object, *, jsonl: bool = False) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")
    if jsonl:
        rows = value
        path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    else:
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def musique_sources(example_id: str) -> set[str]:
    base = example_id.split(":")[2]
    return set(base.split("__", 1)[1].split("_"))


def normalize_question(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def qasper_raw_questions(path: Path) -> dict[str, tuple[str, str]]:
    raw = read_json(path)
    return {
        qa["question_id"]: (paper_id, qa["question"])
        for paper_id, paper in raw.items()
        for qa in paper["qas"]
    }


def prior_exposure_exclusions() -> dict:
    exposed = set(read_json(EXPOSED_TRAIN)["ids"])
    qasper_train = qasper_raw_questions(ROOT / "datasets/raw/qasper/data/qasper-train-v0.3.json")
    exposed_qasper_ids = {
        item.rsplit(":", 1)[-1] for item in exposed if item.startswith("qasper:train:")
    }
    exposed_qasper_normalized = {
        normalize_question(qasper_train[item][1]) for item in exposed_qasper_ids
    }
    pilot_musique_groups: set[str] = set()
    pattern = re.compile(r"musique_full:validation:([A-Za-z0-9_]+)")
    for path in PRIOR_OUTCOME_ARTIFACTS:
        if path.is_file():
            pilot_musique_groups.update(pattern.findall(path.read_text(encoding="utf-8", errors="ignore")))
    return {
        "exposed_qasper_normalized_questions": exposed_qasper_normalized,
        "pilot_musique_source_groups": pilot_musique_groups,
        "prior_outcome_artifacts": [
            str(path.relative_to(ROOT)).replace("\\", "/") for path in PRIOR_OUTCOME_ARTIFACTS if path.is_file()
        ],
    }


def verify_smoke() -> dict:
    validation = read_json(SMOKE_VALIDATION)
    protocol = read_json(SMOKE_PROTOCOL)
    manifest = read_json(SMOKE_MANIFEST)
    answers = load_jsonl(SMOKE_ANSWERS)
    attempts = load_jsonl(SMOKE_ATTEMPTS)
    protocol_body = {key: value for key, value in protocol.items() if key != "protocol_fingerprint"}
    manifest_body = {key: value for key, value in manifest.items() if key != "manifest_fingerprint"}
    checks = {
        "validation_status_passed": validation["status"] == "PASSED_OPERATIONAL_SMOKE_V1_1",
        "reported_first_attempt_valid_48": validation["validity"]["first_attempt_valid_pairs"] == 48,
        "answer_records_48": len(answers) == 48,
        "attempt_records_48": len(attempts) == 48,
        "all_answers_first_attempt": all(row["attempt_for_pair"] == 1 and row["first_attempt_valid"] for row in answers),
        "all_attempts_valid": all(row["attempt_for_pair"] == 1 and row["response_valid"] for row in attempts),
        "protocol_fingerprint_valid": object_sha256(protocol_body) == protocol["protocol_fingerprint"],
        "manifest_fingerprint_valid": object_sha256(manifest_body) == manifest["manifest_fingerprint"],
        "protocol_manifest_link_valid": manifest["protocol_fingerprint"] == protocol["protocol_fingerprint"],
        "validation_artifact_hashes_valid": (
            file_sha256(SMOKE_ANSWERS) == validation["artifacts"]["answers_sha256"]
            and file_sha256(SMOKE_ATTEMPTS) == validation["artifacts"]["attempt_ledger_sha256"]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"smoke v1.1 verification failed: {checks}")
    return {
        "status": "VERIFIED_48_OF_48_FIRST_ATTEMPT_VALID",
        "checks": checks,
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "answers_sha256": file_sha256(SMOKE_ANSWERS),
        "attempt_ledger_sha256": file_sha256(SMOKE_ATTEMPTS),
    }


def build_generation_projection() -> tuple[list[dict], list[dict], dict, dict]:
    exclusions = prior_exposure_exclusions()
    questions: list[dict] = []
    documents_by_key: dict[str, dict] = {}
    dataset_counts: Counter[str] = Counter()
    qasper_papers: set[str] = set()
    musique_groups: set[str] = set()
    excluded_qasper: list[str] = []
    excluded_musique: list[str] = []
    for path, expected_dataset in ((QASPER, "qasper"), (MUSIQUE, "musique_full")):
        source_file_hash = file_sha256(path)
        with path.open(encoding="utf-8") as stream:
            for source_line, line in enumerate(stream, 1):
                row = json.loads(line)
                # Gold-bearing keys are intentionally never indexed here.
                example_id = str(row["example_id"])
                query = str(row["query"])
                dataset = str(row["dataset"])
                if dataset != expected_dataset:
                    raise RuntimeError(f"unexpected dataset in {path}: {dataset}")
                if dataset == "qasper":
                    if normalize_question(query) in exclusions["exposed_qasper_normalized_questions"]:
                        excluded_qasper.append(example_id)
                        continue
                    dependency_group = f"qasper-paper:{row['metadata']['paper_id']}"
                    document_key = dependency_group
                    qasper_papers.add(dependency_group)
                else:
                    source_id = str(row["metadata"]["source_id"])
                    if source_id in exclusions["pilot_musique_source_groups"]:
                        excluded_musique.append(example_id)
                        continue
                    dependency_group = f"musique-group:{source_id}"
                    document_key = f"musique-example:{example_id}"
                    musique_groups.add(dependency_group)
                documents = row["documents"]
                candidate = {
                    "document_key": document_key,
                    "dataset": dataset,
                    "documents": documents,
                }
                existing = documents_by_key.get(document_key)
                if existing is not None and object_sha256(existing) != object_sha256(candidate):
                    raise RuntimeError(f"document key collision with different content: {document_key}")
                documents_by_key[document_key] = candidate
                questions.append({
                    "dataset": dataset,
                    "split": str(row["split"]),
                    "question_id": example_id,
                    "query": query,
                    "query_sha256": sha256(query.encode("utf-8")).hexdigest(),
                    "document_key": document_key,
                    "dependency_group": dependency_group,
                    "source_file": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "source_file_sha256": source_file_hash,
                    "source_line": source_line,
                })
                dataset_counts[dataset] += 1
    questions.sort(key=lambda row: (row["dataset"], row["question_id"]))
    documents = sorted(documents_by_key.values(), key=lambda row: row["document_key"])
    if dataset_counts != Counter({"musique_full": 4832, "qasper": 1408}):
        raise RuntimeError(f"unexpected eligible counts: {dataset_counts}")
    exclusion_report = {
        "qasper_exact_normalized_question_match_to_exposed_training": sorted(excluded_qasper),
        "qasper_excluded_count": len(excluded_qasper),
        "musique_prior_pilot_dependency_groups": sorted(exclusions["pilot_musique_source_groups"]),
        "musique_excluded_full_group_question_ids": sorted(excluded_musique),
        "musique_excluded_count": len(excluded_musique),
        "prior_outcome_artifacts": exclusions["prior_outcome_artifacts"],
    }
    return questions, documents, {
        "question_count": len(questions),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "qasper_document_clusters": len(qasper_papers),
        "musique_dependency_groups": len(musique_groups),
        "document_records": len(documents),
    }, exclusion_report


def qasper_question_to_paper(path: Path) -> dict[str, str]:
    return {key: value[0] for key, value in qasper_raw_questions(path).items()}


def exposure_report(questions: list[dict], counts: dict, exclusions: dict) -> dict:
    final_ids = {row["question_id"] for row in questions}
    exposed_train = set(read_json(EXPOSED_TRAIN)["ids"])
    id_registry = set(read_json(HELDOUT_REGISTRY)["ids"])
    all_registry = read_json(EXPOSED_ALL)
    qasper_train_map = qasper_question_to_paper(ROOT / "datasets/raw/qasper/data/qasper-train-v0.3.json")
    qasper_dev_map = qasper_question_to_paper(ROOT / "datasets/raw/qasper/data/qasper-dev-v0.3.json")
    qasper_test_map = qasper_question_to_paper(ROOT / "datasets/raw/qasper/data/qasper-test-v0.3.json")
    exposed_qasper_questions = {
        item.rsplit(":", 1)[-1] for item in exposed_train if item.startswith("qasper:train:")
    }
    exposed_qasper_papers = {qasper_train_map[item] for item in exposed_qasper_questions}
    final_qasper_papers = set(qasper_test_map.values())
    exposed_musique_sources = {
        source for item in exposed_train if item.startswith("musique_full:train:")
        for source in musique_sources(item)
    }
    final_musique_ids = {item for item in final_ids if item.startswith("musique_full:validation:")}
    final_musique_sources = {source for item in final_musique_ids for source in musique_sources(item)}
    return {
        "status": "ELIGIBLE_WITH_ID_ONLY_PRIOR_ENUMERATION_DISCLOSED",
        "rule": "exclude calibration, tuning, selection, diagnostic, and smoke questions plus dependent document/source groups",
        "final_question_count": len(final_ids),
        "exact_question_id_overlap_with_exposed_training": len(final_ids & exposed_train),
        "prior_id_only_registry_overlap": len(final_ids & id_registry),
        "prior_id_only_registry_implication": (
            "All final IDs were enumerated to establish an exclusion boundary. ID enumeration alone did not expose question text, answers, evidence labels, or outcomes, but the IDs are not literally unseen."
        ),
        "excluded_prior_exposure": exclusions,
        "saved_result_artifacts_with_final_question_outcomes_after_exclusion": [],
        "qasper": {
            "intended_split": "official test",
            "eligible": True,
            "final_questions": counts["dataset_counts"]["qasper"],
            "final_papers": counts["qasper_document_clusters"],
            "exposed_training_questions": len(exposed_qasper_questions),
            "exposed_training_papers": len(exposed_qasper_papers),
            "training_to_test_paper_overlap": len(exposed_qasper_papers & final_qasper_papers),
            "official_train_test_question_overlap": len(set(qasper_train_map) & set(qasper_test_map)),
            "official_dev_test_paper_overlap": len(set(qasper_dev_map.values()) & final_qasper_papers),
            "excluded_exact_normalized_question_duplicates": exclusions["qasper_excluded_count"],
            "uncertainty_cluster": "paper_id",
            "implication": "Eligible for test-split claims; cluster uncertainty by paper. Public-benchmark pretraining exposure remains an unmeasured external limitation.",
        },
        "musique_full": {
            "intended_split": "official development/validation",
            "eligible": True,
            "final_questions": counts["dataset_counts"]["musique_full"],
            "final_dependency_groups": counts["musique_dependency_groups"],
            "exposed_training_source_ids": len(exposed_musique_sources),
            "final_source_ids": len(final_musique_sources),
            "training_to_development_source_id_overlap": len(exposed_musique_sources & final_musique_sources),
            "excluded_prior_pilot_source_groups": len(exclusions["musique_prior_pilot_dependency_groups"]),
            "excluded_prior_pilot_questions": exclusions["musique_excluded_count"],
            "uncertainty_cluster": "source_id (answerable/insufficient variants and shared source-question group)",
            "implication": "Eligible for development-split evaluation, but must be described as development evidence rather than a hidden test result. Public-benchmark pretraining exposure remains unmeasured.",
        },
        "source_registry": {
            "design_exposed_registry_status": all_registry["status"],
            "design_exposed_registry_count": all_registry["count"],
            "heldout_id_registry_sha256": file_sha256(HELDOUT_REGISTRY),
            "exposed_train_registry_sha256": file_sha256(EXPOSED_TRAIN),
        },
        "gold_access_during_this_preparation": False,
    }


def cost_plan(question_count: int) -> dict:
    initial = question_count * len(SYSTEMS)
    maximum = initial * MAX_PER_PAIR
    average_input = SMOKE_INPUT_TOKENS / SMOKE_PAIRS
    average_output = SMOKE_OUTPUT_TOKENS / SMOKE_PAIRS
    estimated_input = average_input * initial
    estimated_output = average_output * initial
    estimated_cost = (
        estimated_input * PRICING["input_per_million"]
        + estimated_output * PRICING["output_per_million"]
    ) / 1_000_000
    per_attempt_ceiling = (
        CONSERVATIVE_INPUT_TOKENS * PRICING["input_per_million"]
        + MAX_COMPLETION_TOKENS * PRICING["output_per_million"]
    ) / 1_000_000
    return {
        "initial_api_attempts": initial,
        "maximum_api_attempts_including_retries": maximum,
        "maximum_attempts_per_pair_including_first": MAX_PER_PAIR,
        "expected_tokens_based_on_v1_1_smoke_means": {
            "input": round(estimated_input),
            "output": round(estimated_output),
            "total": round(estimated_input + estimated_output),
            "basis": "48-pair successful v1.1 smoke; planning estimate, not a guarantee",
        },
        "estimated_initial_cost_usd": round(estimated_cost, 8),
        "conservative_per_attempt": {
            "input_tokens": CONSERVATIVE_INPUT_TOKENS,
            "output_tokens": MAX_COMPLETION_TOKENS,
            "cost_usd": round(per_attempt_ceiling, 8),
        },
        "conservative_initial_round_cost_usd": round(initial * per_attempt_ceiling, 8),
        "conservative_maximum_cost_usd": round(maximum * per_attempt_ceiling, 8),
        "pricing": PRICING,
        "actual_invoice_note": "Actual invoiced charges are unavailable before execution and may differ from estimates; verify provider billing after execution.",
        "runtime": {
            "minimum_seconds_between_attempt_starts": MINIMUM_INTERVAL_SECONDS,
            "initial_round_serial_hours_at_frozen_pacing": initial * MINIMUM_INTERVAL_SECONDS / 3600,
            "maximum_serial_hours_at_frozen_pacing": maximum * MINIMUM_INTERVAL_SECONDS / 3600,
            "base_free_plan_rpd_scenario_days_initial": initial / 1000,
            "base_free_plan_rpd_scenario_days_maximum": maximum / 1000,
            "note": "Exact organization limits must be checked in Groq settings before authorization. The 30-second pacing policy is more conservative than the documented RPM limits but a 1,000-RPD account limit would dominate elapsed time.",
        },
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    smoke = verify_smoke()
    questions, documents, counts, exclusions = build_generation_projection()
    exposure = exposure_report(questions, counts, exclusions)
    costs = cost_plan(len(questions))

    questions_path = OUT / "GENERATION_QUESTIONS.jsonl"
    documents_path = OUT / "GENERATION_DOCUMENTS.jsonl"
    write_new(questions_path, questions, jsonl=True)
    write_new(documents_path, documents, jsonl=True)

    original_protocol = read_json(ORIGINAL_PROTOCOL)
    v1_manifest = read_json(V1)
    v2_manifest = read_json(V2)
    protocol = {
        "status": "FROZEN_AWAITING_SEPARATE_HELDOUT_AUTHORIZATION",
        "protocol_version": "sar-rag-final-answer-evaluation-1.1.0",
        "amends_protocol": {
            "path": str(ORIGINAL_PROTOCOL.relative_to(ROOT)).replace("\\", "/"),
            "sha256": file_sha256(ORIGINAL_PROTOCOL),
            "version": original_protocol["protocol_version"],
            "preserved": True,
        },
        "scope": "held-out retrieval and generation followed by separately gated scoring; no tuning or selective reruns",
        "smoke_prerequisite": smoke,
        "systems": {
            "traditional_k10": original_protocol["systems"]["traditional_k10"],
            "sar_rag_v1": {
                **original_protocol["systems"]["sar_rag_v1"],
                "manifest_sha256_reverified": file_sha256(V1),
                "dynamic_final_k_preserved": True,
            },
            "adaptive_single_view_v2": {
                **original_protocol["systems"]["adaptive_single_view_v2"],
                "manifest_sha256_reverified": file_sha256(V2),
                "dynamic_final_k_preserved": True,
            },
        },
        "retrieval_freeze": {
            "v1_selected_configuration": v1_manifest["frozen_configuration"]["proposed"],
            "v2_selected_configuration": v2_manifest["selected_configuration"],
            "no_retrieval_parameter_changes": True,
            "common_generation_evidence_cap_tokens": 1600,
            "packing_policy": "deterministically traverse native ranked evidence; include a complete chunk only if it fits; never truncate a chunk or refill by changing retrieval",
            "required_separate_logs": [
                "native retrieved evidence IDs/order/text hashes",
                "native retrieval context tokens and dynamic final k",
                "actually supplied evidence IDs/order/text hashes",
                "actually supplied evidence tokens and packing exclusions",
            ],
            "guard": "packing is post-retrieval and may reduce supplied evidence only to enforce the common cap; native retrieval outputs and final k remain the estimand and must not be overwritten",
        },
        "generation": {
            **generation_config_v1_1(),
            "prompt": USER_TEMPLATE_VERSION,
            "schema": SCHEMA_NAME,
            "strict": STRICT,
            "maximum_completion_tokens": MAX_COMPLETION_TOKENS,
            "same_for_every_system": True,
            "outgoing_fields": ["model", "messages", "temperature", "top_p", "max_completion_tokens", "reasoning_effort", "response_format"],
            "settings": {"model": MODEL, "temperature": TEMPERATURE, "top_p": TOP_P, "reasoning_effort": REASONING_EFFORT},
        },
        "baseline_fairness": {
            "predefined_baseline": "traditional_k10",
            "additional_generation_baselines": [],
            "additional_baseline_calls": 0,
            "analysis": "report paired quality differences alongside native retrieved and actually supplied context tokens; report quality-versus-token scatter/paired deltas and Pareto dominance without selecting a post-hoc baseline",
            "actual_consumption": "provider input/output/total billed tokens, supplied evidence tokens, latency, and estimated cost are reported per pair and aggregate",
        },
        "metrics": {
            "answer": ["normalized token F1", "Exact Match"],
            "evidence": [
                "unique gold-evidence recall for native retrieved evidence",
                "unique gold-evidence recall for actually supplied evidence",
                "answerable-only complete-evidence recovery",
                "residual-support recovery reported separately for insufficient questions",
            ],
            "answerability_and_abstention": ["accuracy", "macro precision/recall/F1", "correct abstention", "incorrect abstention", "false-answer rate on insufficient"],
            "efficiency": ["native context tokens", "supplied context tokens", "provider input/output/total billed tokens", "latency", "estimated cost", "dynamic final-k distribution"],
            "faithfulness_separate": {
                "automatic_diagnostics": ["citation ID validity", "claim citation coverage", "cited-evidence lexical support"],
                "interpretation": "diagnostics are not semantic faithfulness proof",
                "judge_model_calls": 0,
                "llm_judge_authorized": False,
                "human_assessment": "separate blinded, preregistered assessment if later authorized; never merged with retrieval/answer correctness metrics",
            },
        },
        "statistics": {
            "comparisons": ["adaptive_single_view_v2 - sar_rag_v1", "sar_rag_v1 - traditional_k10", "adaptive_single_view_v2 - traditional_k10"],
            "confidence_intervals": "10,000-replicate paired cluster bootstrap, 95%, seed 20260926",
            "tests": {
                "continuous": "paired cluster sign-randomization, 10,000 permutations",
                "binary": "exact McNemar when independent pairs apply; cluster randomization for QASPER papers and MuSiQue source groups",
            },
            "clusters": {"qasper": "paper_id", "musique_full": "source_id"},
            "multiplicity": "Holm correction across the three system comparisons separately within each dataset-by-declared-primary-metric family",
            "practical_effect_margins": original_protocol["statistics"]["practical_margins"],
            "required_reporting": ["dataset-specific results", "effect size", "95% CI", "raw p", "Holm-adjusted p", "valid sample count", "cluster count", "answerable/insufficient subgroup counts"],
            "no_cross_dataset_overall_quality_average": True,
        },
        "failure_policy": {
            "maximum_attempts_per_pair_including_first": MAX_PER_PAIR,
            "maximum_retries_per_pair": MAX_PER_PAIR - 1,
            "maximum_total_attempts_including_retries": costs["maximum_api_attempts_including_retries"],
            "round_robin": "finish attempt round N for all unresolved pairs before round N+1",
            "retryable": ["HTTP 429", "HTTP 5xx", "timeout/network error", "json_validate_failed", "JSON parse failure", "local semantic validation failure"],
            "fatal_run_errors": ["authentication/authorization error", "non-output request rejection", "protocol/manifest/request fingerprint drift", "attempt or cost guard mismatch"],
            "rate_limit": {"minimum_seconds_between_attempt_starts": MINIMUM_INTERVAL_SECONDS, "honor_retry_after": True, "retry_after_safety_margin_seconds": 0.25},
            "validity_reporting": ["first-attempt validity", "eventual validity", "attempt of first valid response"],
            "persistent_generation_failure_scoring": {
                "retained_in_denominator": True,
                "answer_token_f1": 0,
                "exact_match": 0,
                "answerability": "incorrect special GENERATION_FAILURE outcome; never counted as abstention",
                "retrieval_metrics": "score from the sealed native/supplied retrieval logs because retrieval completed before generation",
                "operational_metrics": "include all failed-attempt token usage, estimated cost, and wall/attempt latency",
            },
            "prohibited": ["silent exclusion", "answer repair", "configuration change", "selective rerun", "answer reuse across systems"],
        },
        "separation_and_scoring_gate": {
            "generation_inputs_contain_gold": False,
            "gold_information_allowed_in_prompt_or_retrieval": False,
            "gold_stage": "open gold answer/evidence labels only after predictions and retrieval logs are complete and cryptographically sealed, under separate scoring authorization",
            "heldout_generation_authorized": False,
            "heldout_scoring_authorized": False,
        },
        "cost_and_attempt_plan": costs,
        "pricing_verification": {
            "model_page": PRICING["source"],
            "rate_limits_page": "https://console.groq.com/docs/rate-limits",
            "observed_official_rates": PRICING,
            "account_specific_limits_unverified_without_provider_settings_access": True,
        },
        "interpretation_limits": [
            "MuSiQue development results are development-split evidence, not hidden-test evidence",
            "QASPER test results support dataset-specific test claims only",
            "public-model benchmark pretraining exposure is unknown",
            "smoke success is operational evidence only and supports no superiority claim",
        ],
    }
    protocol["protocol_fingerprint"] = object_sha256(protocol)
    protocol_path = OUT / "FINAL_EVALUATION_PROTOCOL_AMENDMENT_V1_1.json"
    write_new(protocol_path, protocol)

    exposure["generation_questions_sha256"] = file_sha256(questions_path)
    exposure["generation_documents_sha256"] = file_sha256(documents_path)
    exposure_path = OUT / "DATASET_ELIGIBILITY_AND_EXPOSURE.json"
    write_new(exposure_path, exposure)

    pairs = [
        {
            "logical_key": f"{row['dataset']}|{row['question_id']}|{system}",
            "dataset": row["dataset"],
            "question_id": row["question_id"],
            "system": system,
            "dependency_group": row["dependency_group"],
            "query_sha256": row["query_sha256"],
        }
        for row in questions for system in SYSTEMS
    ]
    manifest = {
        "status": "FROZEN_AWAITING_OFFLINE_RETRIEVAL_INPUT_BUILD_AND_SEPARATE_PAID_AUTHORIZATION",
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "systems": list(SYSTEMS),
        "question_count": len(questions),
        "pair_count": len(pairs),
        "dataset_question_counts": counts["dataset_counts"],
        "generation_questions_sha256": file_sha256(questions_path),
        "generation_documents_sha256": file_sha256(documents_path),
        "exposure_report_sha256": file_sha256(exposure_path),
        "frozen_retrieval_manifest_sha256": {"sar_rag_v1": file_sha256(V1), "adaptive_single_view_v2": file_sha256(V2)},
        "initial_api_attempts": costs["initial_api_attempts"],
        "maximum_total_api_attempts_including_retries": costs["maximum_api_attempts_including_retries"],
        "maximum_attempts_per_pair_including_first": MAX_PER_PAIR,
        "maximum_cost_usd": costs["conservative_maximum_cost_usd"],
        "pairs": pairs,
    }
    manifest["manifest_fingerprint"] = object_sha256(manifest)
    manifest_path = OUT / "FROZEN_QUESTION_SYSTEM_MANIFEST.json"
    write_new(manifest_path, manifest)

    authorization = {
        "status": "DISABLED_PROPOSAL_NOT_AUTHORIZATION",
        "authorized_stage": "heldout_generation_v1_1_only",
        "user_authorized": False,
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "run_manifest_fingerprint": manifest["manifest_fingerprint"],
        "request_manifest_fingerprint": "MUST_BE_FILLED_AFTER_OFFLINE_RETRIEVAL_REQUEST_FREEZE",
        "maximum_total_api_attempts_including_retries": costs["maximum_api_attempts_including_retries"],
        "maximum_attempts_per_pair_including_first": MAX_PER_PAIR,
        "maximum_cost_usd": costs["conservative_maximum_cost_usd"],
        "scope_note": "Does not authorize scoring, judge calls, configuration changes, or calls beyond the exact frozen pair manifest.",
    }
    write_new(OUT / "PROPOSED_AUTHORIZATION_DISABLED.json", authorization)

    package = {
        "status": "PREPARED_NO_API_CALLS_NO_GOLD_LABEL_ACCESS",
        "smoke": smoke,
        "counts": counts,
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "artifacts": {
            path.name: file_sha256(path)
            for path in (questions_path, documents_path, protocol_path, exposure_path, manifest_path, OUT / "PROPOSED_AUTHORIZATION_DISABLED.json")
        },
        "cost_and_attempt_plan": costs,
        "api_calls": 0,
        "gold_labels_accessed": False,
    }
    write_new(OUT / "PREPARATION_VALIDATION.json", package)
    print(json.dumps(package, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
