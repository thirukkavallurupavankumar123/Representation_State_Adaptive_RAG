"""Audit Phase E and freeze the answer-generation evaluation before paid use."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sar_rag.final_evaluation import file_sha256, object_sha256  # noqa: E402

OUT = ROOT / "output" / "final_evaluation"
RUN = ROOT / "output" / "adaptive_single_view_v2" / "recalibration_seed20260905"
SCREENING = ROOT / "output" / "independent_verification" / "screening_v3_n286" / "SCREENING_V2_PER_EXAMPLE.csv"
V1 = ROOT / "output" / "calibration" / "frozen_manifest.json"
V2 = RUN / "FROZEN_V2_MANIFEST.json"

MODEL = "openai/gpt-oss-120b"
SMOKE_N = 16
SYSTEMS = ("traditional_k10", "sar_rag_v1", "adaptive_single_view_v2")
PRICING = {"input_per_million": 0.15, "cached_input_per_million": 0.075, "output_per_million": 0.60}


def configured_model_only() -> tuple[str, str]:
    env_path = ROOT / ".env"
    if env_path.exists():
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                if key.strip() == "GROQ_MODEL":
                    return value.strip().strip("\"'") or MODEL, ".env:GROQ_MODEL"
    return MODEL, "code_default"


def mean(rows: list[dict], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows)


def phase_e_audit() -> dict:
    with (RUN / "SELECTION_PER_EXAMPLE.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    selected_id = json.loads(V2.read_text(encoding="utf-8"))["selected_configuration_id"]
    v2 = [row for row in rows if row["mode"] == "adaptive" and row["configuration_id"] == selected_id]
    v1 = [row for row in rows if row["mode"] == "v1_full"]
    if len(v2) != 63 or len(v1) != 63:
        raise RuntimeError("expected 63 paired v1/v2 selection rows")
    if {row["example_id"] for row in v2} != {row["example_id"] for row in v1}:
        raise RuntimeError("v1/v2 selection IDs are not paired")
    by_id_v1 = {row["example_id"]: row for row in v1}
    recall_diffs = [float(row["evidence_recall"]) - float(by_id_v1[row["example_id"]]["evidence_recall"]) for row in v2]
    token_diffs = [float(row["context_tokens"]) - float(by_id_v1[row["example_id"]]["context_tokens"]) for row in v2]
    return {
        "status": "RECOMPUTED_FROM_SAVED_PER_EXAMPLE_ROWS",
        "paired_n": 63,
        "v2_evidence_recall": mean(v2, "evidence_recall"),
        "v2_mean_actual_context_tokens": mean(v2, "context_tokens"),
        "v1_evidence_recall": mean(v1, "evidence_recall"),
        "v1_mean_actual_context_tokens": mean(v1, "context_tokens"),
        "mean_paired_v2_minus_v1_recall": sum(recall_diffs) / len(recall_diffs),
        "mean_paired_v2_minus_v1_tokens": sum(token_diffs) / len(token_diffs),
        "v2_point_dominated_by_v1": mean(v2, "evidence_recall") < mean(v1, "evidence_recall") and mean(v2, "context_tokens") > mean(v1, "context_tokens"),
        "source_sha256": file_sha256(RUN / "SELECTION_PER_EXAMPLE.csv"),
        "frozen_v1_sha256": file_sha256(V1),
        "frozen_v2_sha256": file_sha256(V2),
    }


def smoke_examples() -> list[dict]:
    with SCREENING.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["arm"] == "sar_rag" and row["condition"] == "full"]
    quotas = {
        "qasper|ANSWERABLE|long": 2, "qasper|ANSWERABLE|medium": 1, "qasper|ANSWERABLE|short": 1,
        "qasper|INSUFFICIENT|long": 2, "qasper|INSUFFICIENT|medium": 1, "qasper|INSUFFICIENT|short": 1,
        "musique_full|ANSWERABLE|2hop": 2, "musique_full|ANSWERABLE|3hop": 1, "musique_full|ANSWERABLE|4hop": 1,
        "musique_full|INSUFFICIENT|2hop": 2, "musique_full|INSUFFICIENT|3hop": 1, "musique_full|INSUFFICIENT|4hop": 1,
    }
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["stratum"]].append(row)
    chosen = []
    for stratum, count in quotas.items():
        ranked = sorted(grouped[stratum], key=lambda row: sha256(f"20260905|{stratum}|{row['example_id']}".encode()).hexdigest())
        if len(ranked) < count:
            raise RuntimeError(f"insufficient design-exposed smoke examples: {stratum}")
        for row in ranked[:count]:
            chosen.append({
                "example_id": row["example_id"], "dataset": row["dataset"],
                "stratum": row["stratum"],
                "answerability": "ANSWERABLE" if row["answerable"] == "1" else "INSUFFICIENT",
                "source_design_partition": row["design_partition"],
                "selection_hash": sha256(f"20260905|{stratum}|{row['example_id']}".encode()).hexdigest(),
            })
    return sorted(chosen, key=lambda row: (row["dataset"], row["stratum"], row["selection_hash"]))


def write_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")
    path.write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model, model_source = configured_model_only()
    if model != MODEL:
        raise RuntimeError(f"configured model differs from reviewed model: {model}")
    audit = phase_e_audit()
    examples = smoke_examples()
    heldout = json.loads((ROOT / "output" / "adaptive_single_view_v2" / "HELDOUT_ID_REGISTRY.json").read_text(encoding="utf-8"))
    heldout_counts = Counter(":".join(item.split(":")[:2]) for item in heldout["ids"])
    final_n = heldout_counts["qasper:test"] + heldout_counts["musique_full:validation"]
    generation = {
        "model": model,
        "model_source": model_source,
        "temperature": 0,
        "top_p": 1,
        "reasoning_effort": "medium",
        "max_output_tokens": 800,
        "response_format": "strict JSON schema",
        "calls_per_system_question_pair": 1,
        "max_retries_after_initial_attempt": 2,
        "retry_backoff_seconds": [1, 2],
        "timeout_seconds": 90,
    }
    protocol = {
        "status": "FROZEN_BEFORE_ANY_PAID_EXECUTION",
        "protocol_version": "sar-rag-final-answer-evaluation-1.0.0",
        "frozen_date": "2026-09-05",
        "scope": "answer generation only after frozen deterministic retrieval; no tuning on smoke or held-out outcomes",
        "phase_e_audit": audit,
        "systems": {
            "traditional_k10": {
                "role": "primary traditional baseline", "representation": "sentence_window",
                "fixed_k": 10, "fixed_k_operating_points_predeclared": [10, 20],
                "reason": "k=10 is the development token-comparable baseline; k=20 remains retrieval-only Pareto context and receives no answer call",
                "embedding": "sentence-transformers/all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
            },
            "sar_rag_v1": {"manifest": "output/calibration/frozen_manifest.json", "manifest_sha256": audit["frozen_v1_sha256"], "recalibration": False},
            "adaptive_single_view_v2": {
                "manifest": "output/adaptive_single_view_v2/recalibration_seed20260905/FROZEN_V2_MANIFEST.json",
                "manifest_sha256": audit["frozen_v2_sha256"], "representation": "paragraph",
                "max_context_tokens": 1600, "relevance_percentile": 0.45, "patience": 1, "recalibration": False,
            },
        },
        "fairness": {
            "identical": ["question set", "embedding model/revision", "answer model", "prompt", "schema", "decoding", "maximum answer input context", "metrics"],
            "common_answer_evidence_cap_tokens": 1600,
            "v1_policy": "run frozen v1 retrieval unchanged, then deterministically pack its ranked evidence to the common 1600-token answer-input cap; report native retrieval tokens and packed answer-input tokens separately",
            "actual_consumption_reported": True, "pareto_analysis": True, "matched_actual_token_analysis": True,
        },
        "generation": generation,
        "cache": {
            "fingerprint_fields": ["dataset", "question_id", "system", "evidence_hash", "prompt_hash", "model", "decoding"],
            "resume": "reuse only exact request and run fingerprints; reject logical-key or checkpoint-configuration mismatch",
            "cross_system_identical_context_reuse": False,
            "reason": "system identity remains in the estimand and fingerprint; each system-question pair receives its own call even when evidence bytes match",
            "failure_policy": "record failure separately; never convert it to an empty answer; retry at most twice, then leave the pair missing",
        },
        "smoke_test": {
            "manifest": "output/final_evaluation/SMOKE_TEST_MANIFEST.json", "questions": SMOKE_N,
            "systems": list(SYSTEMS), "maximum_calls": SMOKE_N * len(SYSTEMS),
            "purpose": "operational validation only; never research evidence",
            "selection": "deterministic stratified hash sample from previously design-exposed training screening data, frozen before response generation",
            "pass_criteria": [
                "strict response schema", "no undeclared prompt/context truncation", "INSUFFICIENT parses correctly",
                "input/output/total token and latency accounting", "bounded retry and checkpoint behavior",
                "exact resume creates zero duplicate calls", "evidence identifiers preserved", "no secrets in artifacts",
                "no configuration drift", "identical answer model/prompt/decoding across systems",
            ],
        },
        "final_heldout": {
            "datasets": {
                "qasper": {"split": "official test", "id_only_count": heldout_counts["qasper:test"]},
                "musique_full": {"split": "official validation/development", "id_only_count": heldout_counts["musique_full:validation"]},
            },
            "question_count": final_n, "systems": list(SYSTEMS), "maximum_answer_calls": final_n * len(SYSTEMS),
            "label_policy": "retrieve and generate predictions first; seal prediction hashes; only then open gold labels for scoring; never tune or rerun from outcomes",
            "dataset_specific_reporting": True,
        },
        "outcomes": {
            "primary": ["answer token F1", "unique gold-evidence recall", "answerable-only complete-evidence recovery", "mean actual context tokens"],
            "secondary": ["Exact Match", "answer precision", "answer recall", "evidence precision", "MRR", "nDCG with granularity caveat", "answerability precision/recall/F1/accuracy", "correct abstention", "false-answer rate on insufficient", "incorrect abstention", "deterministic citation validity/coverage", "answer relevancy", "input/output/total tokens", "latency mean/median/SD/p95", "cost per question/total", "final-k distribution", "retrieval stop reasons"],
        },
        "faithfulness_relevancy": {
            "primary_free_measurements": ["citation ID validity", "fraction of claims with citations", "cited-evidence lexical support diagnostic", "answer-question token overlap diagnostic"],
            "limitation": "lexical diagnostics are not semantic faithfulness proof",
            "manual_validation": "60 prediction triples, stratified and selected by frozen hash after predictions are sealed",
            "optional_llm_judge": {
                "authorized": False, "separate_stage": True, "model": model,
                "blind_system_names": True, "randomize_answer_order_by_hash": True,
                "one_comparative_call_per_question": True, "maximum_calls": final_n,
                "requires_separate_cost estimate and explicit authorization": True,
            },
        },
        "statistics": {
            "comparisons": ["v2-v1", "v1-traditional_k10", "v2-traditional_k10"],
            "methods": ["paired bootstrap 95% CI", "paired randomization", "exact McNemar for binary outcomes"],
            "multiplicity": "Holm correction across the three comparisons separately for each declared primary outcome",
            "report": ["valid paired n", "dataset-specific results", "answerable/insufficient subgroups", "effect sizes", "raw and Holm-adjusted p-values"],
            "practical_margins": {"answer_F1_gain": 0.02, "answer_F1_noninferiority": -0.02, "evidence_recall_noninferiority": -0.02, "answerable_CER_noninferiority": -0.03, "meaningful_token_latency_or_cost_reduction_percent": 10},
        },
        "interpretation": {
            "outperforms": "answer-F1 paired CI excludes zero and mean gain >=0.02, with evidence recall/CER above frozen noninferiority margins and no unacceptable cost regression",
            "efficiency improvement": "answer-F1 lower CI >=-0.02 and at least 10% reduction in tokens, latency, or cost",
            "pareto competitive": "no compared system improves a declared quality-efficiency dimension without worsening another",
            "otherwise": "unsupported",
            "development_caveat": "retrieval-only development and smoke results are never held-out answer-quality evidence",
        },
        "authorization": {
            "paid_smoke": "requires a separate matching PAID_SMOKE_AUTHORIZATION.json and --execute-paid after explicit user authorization",
            "full_heldout": "requires smoke pass, sealed smoke report, separately matching held-out authorization, frozen cost ceiling, and no protocol drift",
        },
        "pricing": {**PRICING, "currency": "USD", "unit": "per 1M tokens", "source": "https://console.groq.com/docs/model/openai/gpt-oss-120b", "verified_date": "2026-09-05"},
    }
    protocol["protocol_fingerprint"] = object_sha256(protocol)
    smoke = {
        "status": "FROZEN_BEFORE_PAID_SMOKE",
        "seed": 20260905, "question_count": len(examples), "systems": list(SYSTEMS),
        "maximum_calls": len(examples) * len(SYSTEMS), "cross_system_call_reuse": False,
        "examples": examples,
        "stratum_counts": dict(sorted(Counter(row["stratum"] for row in examples).items())),
        "dataset_counts": dict(sorted(Counter(row["dataset"] for row in examples).items())),
        "answerability_counts": dict(sorted(Counter(row["answerability"] for row in examples).items())),
        "source_sha256": file_sha256(SCREENING), "protocol_fingerprint": protocol["protocol_fingerprint"],
    }
    smoke["manifest_fingerprint"] = object_sha256(smoke)

    write_new(OUT / "PHASE_E_RESULT_AUDIT.json", json.dumps(audit, indent=2, sort_keys=True) + "\n")
    write_new(OUT / "FINAL_EVALUATION_PROTOCOL.json", json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    write_new(OUT / "SMOKE_TEST_MANIFEST.json", json.dumps(smoke, indent=2, sort_keys=True) + "\n")

    md = f"""# Frozen final answer-generation evaluation protocol

Status: **FROZEN BEFORE ANY PAID EXECUTION**  
Protocol fingerprint: `{protocol['protocol_fingerprint']}`

The study compares Traditional sentence-window RAG at frozen k=10, unchanged frozen SAR-RAG v1, and frozen paragraph-based Adaptive Single-View v2. All use `{model}`, the same strict evidence-only prompt/schema, temperature 0, top-p 1, medium reasoning, an 800-token output cap, and a common 1,600-token packed evidence cap. Traditional k=20 remains a declared retrieval-only Pareto comparator and receives no answer call.

The paid smoke test contains 16 already design-exposed training questions: eight QASPER and eight MuSiQue; eight answerable and eight insufficient. It requires exactly 48 calls because cross-system response sharing is prohibited. It is operational validation only.

Final held-out scoring uses QASPER official test ({heldout_counts['qasper:test']} ID-only records) and MuSiQue-Full validation ({heldout_counts['musique_full:validation']} ID-only records), for {final_n} questions and at most {final_n * 3} answer calls. Predictions must be sealed before labels are opened. No final outcome may change retrieval, prompt, decoding, or analysis rules.

Primary outcomes are answer token F1, unique gold-evidence recall, answerable-only complete-evidence recovery, and actual context tokens. Paired bootstrap intervals, paired randomization tests, exact McNemar tests, and Holm correction are frozen in the JSON protocol, along with practical margins and interpretation rules.

No paid call and no held-out label access occurred while creating this protocol.
"""
    write_new(OUT / "FINAL_EVALUATION_PROTOCOL.md", md)

    expected_input, expected_output = 2100, 250
    max_input, max_output = 2800, generation["max_output_tokens"]
    smoke_expected = SMOKE_N * (expected_input * PRICING["input_per_million"] + expected_output * PRICING["output_per_million"]) / 1_000_000
    smoke_max = SMOKE_N * 3 * (max_input * PRICING["input_per_million"] + max_output * PRICING["output_per_million"]) / 1_000_000
    # Correct expected cost includes all three systems.
    smoke_expected *= 3
    full_expected = final_n * 3 * (expected_input * PRICING["input_per_million"] + expected_output * PRICING["output_per_million"]) / 1_000_000
    full_max = final_n * 3 * (max_input * PRICING["input_per_million"] + max_output * PRICING["output_per_million"]) / 1_000_000
    cost_md = f"""# Expected calls and cost

Pricing frozen on 2026-09-05 from Groq's official model page for `{model}`: $0.15/M input tokens, $0.075/M cached input tokens, and $0.60/M output tokens. Estimates conservatively assume no cache discount.

| Stage | Questions | Systems | Calls | Expected tokens/call (in/out) | Hard planning ceiling (in/out) | Expected cost | Ceiling cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Paid smoke | 16 | 3 | 48 | 2,100 / 250 | 2,800 / 800 | ${smoke_expected:.5f} | ${smoke_max:.5f} |
| Final held-out answers | {final_n} | 3 | {final_n * 3} | 2,100 / 250 | 2,800 / 800 | ${full_expected:.2f} | ${full_max:.2f} |

The 2,800-input figure is an authorization/cost planning ceiling, not an instruction to pad prompts. Actual billed usage, latency, retries, and cost are recorded per call. Any prompt exceeding the declared ceiling fails before submission; it is not silently truncated.

The optional blinded comparative LLM-judge stage is separate, has at most {final_n} additional calls, and remains unauthorized. Its cost must be re-estimated at the time of a separate authorization.
"""
    write_new(OUT / "EXPECTED_CALLS_AND_COST.md", cost_md)

    gate = f"""# Authorization gate

No paid operation is authorized by this file. Before the first Groq request, all of the following must hold:

1. The user explicitly authorizes the 48-call paid smoke test.
2. `PAID_SMOKE_AUTHORIZATION.json` is created after that authorization and matches protocol fingerprint `{protocol['protocol_fingerprint']}` and smoke manifest fingerprint `{smoke['manifest_fingerprint']}`.
3. Its maximum calls is exactly 48 and its maximum estimated charge is no more than ${smoke_max:.5f} under the frozen rates.
4. The prepared request file passes fingerprint and schema validation.
5. The command includes the deliberate `--execute-paid` flag.

The full held-out run and optional LLM judge each require a later, separate authorization. Smoke authorization never authorizes either stage.
"""
    write_new(OUT / "AUTHORIZATION_GATE.md", gate)
    commands = """# Reproduction commands

```powershell
& '.venv\\Scripts\\python.exe' 'tools/freeze_final_evaluation_protocol.py'
& '.venv\\Scripts\\python.exe' 'tools/prepare_smoke_test_requests.py'
& '.venv\\Scripts\\python.exe' 'tools/run_final_answer_evaluation.py' --mode dry-run --protocol 'output/final_evaluation/FINAL_EVALUATION_PROTOCOL.json' --manifest 'output/final_evaluation/SMOKE_TEST_MANIFEST.json' --checkpoint-dir 'output/final_evaluation/dry_run_checkpoint'
& '.venv\\Scripts\\python.exe' -m pytest -q
```

The freeze command intentionally refuses to overwrite frozen files. The paid command is documented in `AUTHORIZATION_GATE.md` and must not run until the separate authorization file exists after explicit user approval.
"""
    write_new(OUT / "REPRODUCTION_COMMANDS.md", commands)
    print(json.dumps({"status": protocol["status"], "protocol_fingerprint": protocol["protocol_fingerprint"], "smoke_questions": len(examples), "smoke_max_calls": 48, "final_max_calls": final_n * 3}))


if __name__ == "__main__":
    main()
