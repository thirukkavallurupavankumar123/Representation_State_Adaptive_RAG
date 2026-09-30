"""Freeze the preregistered Adaptive Single-View Retrieval V2 protocol.

This is a one-way creation command: it refuses to overwrite an existing frozen
protocol.  It scans only existing research artifacts and normalized samples for
design-exposed IDs; official held-out source files are never opened.
"""
from __future__ import annotations

import csv
import hashlib
import json
import pathlib
from collections import Counter
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "adaptive_single_view_v2"
PROTOCOL_JSON = OUT / "PROTOCOL.json"


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_json_ids(value: Any, found: set[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "example_id" and isinstance(item, str):
                found.add(item)
            elif key in {"sample_ids", "processed_ids", "example_ids"} and isinstance(item, list):
                found.update(str(v) for v in item if isinstance(v, str))
            collect_json_ids(item, found)
    elif isinstance(value, list):
        for item in value:
            collect_json_ids(item, found)


def design_exposed_ids() -> tuple[list[str], list[str]]:
    found: set[str] = set()
    sources: list[str] = []
    roots = (ROOT / "output" / "calibration", ROOT / "output" / "independent_verification",
             ROOT / "output" / "q2_analysis_pilot", ROOT / "output" / "pilot_frozen_qasper2",
             ROOT / "output" / "pilot_fair_generator_qasper2", ROOT / "datasets" / "normalized")
    for base in roots:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".csv", ".json", ".jsonl"}:
                continue
            try:
                before = len(found)
                if path.suffix.lower() == ".csv":
                    with path.open(encoding="utf-8-sig", newline="") as stream:
                        reader = csv.DictReader(stream)
                        if "example_id" not in (reader.fieldnames or []):
                            continue
                        found.update(row["example_id"] for row in reader if row.get("example_id"))
                elif path.suffix.lower() == ".jsonl":
                    with path.open(encoding="utf-8") as stream:
                        for line in stream:
                            if line.strip():
                                collect_json_ids(json.loads(line), found)
                else:
                    collect_json_ids(json.loads(path.read_text(encoding="utf-8")), found)
                if len(found) > before:
                    sources.append(path.relative_to(ROOT).as_posix())
            except (UnicodeDecodeError, csv.Error, json.JSONDecodeError, OSError):
                continue
    return sorted(found), sources


def protocol(exposed_hash: str, exposed_count: int) -> dict:
    return {
        "status": "FROZEN_BEFORE_IMPLEMENTATION",
        "protocol_version": "adaptive-single-view-v2.0.0",
        "frozen_date": "2026-09-05",
        "scope": "training-only retrieval recalibration; no LLM calls; no held-out access",
        "research_question": (
            "Can a deterministically configured adaptive single-view retriever preserve retrieval quality "
            "while improving actual context efficiency relative to declared fixed-k and shallow-probe comparators?"
        ),
        "hypotheses": {
            "H1_retrieval": "At least one preregistered single-view operating point improves unique gold-span recall by >=0.02 over the shallow-probe comparator.",
            "H2_efficiency": "At recall noninferiority margin -0.02, adaptive retrieval reduces mean actual context tokens by >=10% relative to a fixed-k comparator at the same safety maximum.",
            "H3_answerable_CER": "The selected method improves answerable-only complete-evidence recovery by >=0.03 over the shallow-probe comparator.",
            "null_safe": "If neither quality nor efficiency criteria are met, the method is reported as Pareto competitive or unsupported; no superiority claim is permitted.",
        },
        "partitions": {
            "design_exposed": {
                "definition": "Every example ID discoverable in prior calibration, pilot, screening, diagnostic, analysis, or normalized-sample artifacts at freeze time.",
                "file": "DESIGN_EXPOSED_IDS.json", "count": exposed_count, "ids_sha256": exposed_hash,
                "use": "method design only; excluded from v2 tuning and selection",
            },
            "development_population": "Official QASPER train and MuSiQue-Full train only, after all exclusions.",
            "v2_tuning": "18 examples per stratum (162 total), assigned by deterministic within-stratum hash; fits relevance threshold and patience.",
            "v2_selection": "7 examples per stratum (63 total), assigned before execution; used once by the frozen selection rule.",
            "final_held_out": {
                "locked": ["QASPER official test", "MuSiQue-Full official development/validation for locally scored confirmation", "MuSiQue-Full official test for hidden-label submission only"],
                "access_during_v2_development": "forbidden",
                "id_disjointness": "v2 IDs must contain the literal split tag ':train:'; QASPER test IDs contain ':test:', proving structural disjointness without opening test records.",
            },
        },
        "sample": {
            "seed": 20260905, "sampling": "streaming bottom-SHA256, deterministic within stratum",
            "per_stratum": 25, "tuning_per_stratum": 18, "selection_per_stratum": 7,
            "expected_strata": 9, "expected_total": 225, "expected_tuning": 162, "expected_selection": 63,
            "strata": [
                "qasper|ANSWERABLE|short", "qasper|ANSWERABLE|medium", "qasper|ANSWERABLE|long",
                "musique_full|ANSWERABLE|2hop", "musique_full|ANSWERABLE|3hop", "musique_full|ANSWERABLE|4hop",
                "musique_full|INSUFFICIENT|2hop", "musique_full|INSUFFICIENT|3hop", "musique_full|INSUFFICIENT|4hop",
            ],
        },
        "inclusion_exclusion": {
            "include": ["official training records", "non-empty query", "non-empty parsed document", "at least one gold evidence span"],
            "exclude": ["all design-exposed IDs", "all frozen v1 calibration IDs", "all prior diagnostic/screening IDs", "any non-train split", "QASPER insufficient questions with zero gold evidence spans", "malformed or duplicate IDs"],
            "answerable": "included; answerable-only CER is primary",
            "insufficient": "MuSiQue insufficient examples with residual gold support are included; residual-support complete recovery is secondary and never mixed with answerable CER",
        },
        "embedding": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
            "local_path": "models/all-MiniLM-L6-v2", "dimension": 384,
            "weight_sha256": "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db",
            "implementation": "Hugging Face Transformers/SentenceTransformers-compatible local snapshot; L2-normalized mean pooling",
        },
        "candidate_representations": ["semantic", "paragraph", "hierarchical_v2", "sentence_window"],
        "primary_method": {
            "representation_scope": "exactly one predeclared run-level representation",
            "per_query_router": False, "rrf": False, "representation_pairs": False,
            "initial_probe_depth": 3, "adaptive_batch_size": 3,
            "relevance": "empirical percentile-normalized cosine relevance, deterministic",
            "retained_stops": ["relevance threshold", "empty-batch/no-gain patience", "token safety maximum", "retrieval-round cap", "candidate cap", "chunk cap"],
            "removed_primary_controls": ["novelty gate", "similarity-gap gate", "marginal-utility gate", "redundancy gate"],
            "deduplication": "after adaptive retrieval and before final context accounting/output; efficiency mechanism only",
            "final_k": "emergent; never selected or fixed as a tuning parameter",
            "llm_policy": "zero LLM calls during selection, retrieval, tuning, or this recalibration; Groq is permitted only for a separately authorized final answer call",
        },
        "token_operating_points": [800, 1200, 1600],
        "token_safety_maximum": 2000,
        "recalibration_grid": {
            "relevance_percentile_threshold": [0.45, 0.55, 0.65, 0.75],
            "empty_batch_patience": [1, 2, 3],
            "max_retrieval_rounds": [6, 8],
            "max_candidates_examined": [30, 60],
            "max_final_chunks": [12, 20],
            "fixed": {"probe_depth": 3, "adaptive_batch_size": 3, "deduplicate_after_retrieval": True},
            "fit_partition": "v2_tuning only",
        },
        "metrics": {
            "primary": ["unique gold-span recall over all evidence-scored questions", "answerable-only complete-evidence recovery", "actual context tokens", "answer token F1 only after paid authorization"],
            "secondary": ["evidence precision", "residual-support complete recovery for insufficient questions", "MRR", "nDCG (reported with chunk-granularity caveat)", "pre/post-dedup k and final k", "latency", "stop-reason distribution"],
        },
        "fair_comparison": {
            "requirements": ["identical query set", "identical embedding model and snapshot", "identical maximum context allowance", "same answer model and prompt in any paid phase", "actual context consumption reported", "recall-versus-token Pareto frontier", "matched actual-token comparisons"],
            "matched_actual_tokens": "mean context tokens within 10% on identical queries; paired token differences also reported",
            "no_full_budget_requirement": "adaptive systems need not consume the 2,000-token safety maximum",
        },
        "inference": {
            "paired_bootstrap_resamples": 20000, "paired_bootstrap_seed": 20260905,
            "paired_randomization_resamples": 20000, "binary_test": "exact McNemar",
            "multiple_testing": "Holm correction across declared primary comparisons within each primary metric",
            "interval": 0.95,
        },
        "practical_effect_margins": {
            "meaningful_recall_gain": 0.02, "recall_noninferiority": -0.02,
            "meaningful_answerable_CER_gain": 0.03, "meaningful_context_reduction_percent": 10.0,
            "meaningful_answer_token_F1_gain_after_authorization": 0.02,
        },
        "frozen_selection_rule": {
            "tuning_stage": "For each representation and token point, retain the parameter tuple with highest tuning recall; treat differences <=0.01 as tied, then maximize answerable CER, minimize actual tokens, minimize candidates, and use canonical tuple order as final tie-break.",
            "selection_stage": "Evaluate only the 12 tuning-selected representation/token candidates. Shortlist candidates within 0.01 of the highest selection recall; then maximize answerable CER, minimize actual tokens, minimize candidates, and break exact ties by the preregistered representation order [semantic, paragraph, hierarchical_v2, sentence_window] followed by ascending token point.",
            "prior_screening_guard": "The representation order is only an exact tie-break. Semantic receives no score bonus and is not selected merely because it led the design-exposed screening sample.",
            "classification": {
                "better_on_retrieval_quality": "recall gain >=0.02 with Holm-adjusted p<0.05, or answerable CER gain >=0.03 with adjusted p<0.05, versus the declared comparator",
                "better_only_on_efficiency": "recall lower CI >=-0.02 and context reduction >=10% with paired token p<0.05, without a quality win",
                "pareto_competitive": "non-dominated in recall versus actual-token space but neither superiority rule passes",
                "unsupported": "dominated or fails both noninferiority and quality criteria",
            },
        },
        "comparators": ["probe-only k=3 for the same representation", "fixed k=10", "fixed k=20", "v1 full SAR-RAG descriptive only"],
        "run_integrity": {
            "required_records": ["exact command", "configuration", "seed", "package versions", "input hashes", "failures and retries", "artifact hashes", "configuration fingerprint", "embedding fingerprint"],
            "resume": "append-only per-example output; reject if protocol, configuration, embedding, input, sample, seed, or code fingerprint differs; never mix configurations",
            "failed_runs": "record failure row and retry count; deterministic retry only under identical fingerprints; otherwise start a new versioned run directory",
        },
        "paid_smoke_test_gate": {
            "condition": "All retrieval-only v2 artifacts complete; full tests pass; frozen manifest reproduces exactly; overlap audit is zero; no unresolved failures; answer prompt/model fingerprints are frozen; user gives explicit paid-run authorization with a call ceiling and cost ceiling.",
            "data": "design-exposed training examples only; never final held-out records",
        },
        "final_held_out_gate": {
            "condition": "Paid smoke passes without method changes; all protocol/manifest/code/prompt/model hashes remain frozen; exact run cost and resume plan are recorded; user gives a separate explicit authorization for final held-out evaluation.",
            "after_access": "No tuning, selection, threshold, representation, prompt, or analysis-rule changes are allowed; deviations require a new protocol and the accessed examples remain exposed.",
        },
    }


def markdown(p: dict) -> str:
    return f"""# Adaptive Single-View Retrieval V2 - preregistered protocol

**Status:** {p['status']}  
**Version:** {p['protocol_version']}  
**Frozen:** {p['frozen_date']}  
**Scope:** training-only retrieval; no LLM calls and no held-out access.

## Research question and hypotheses

The study asks whether a deterministically configured adaptive single-view retriever can preserve or improve evidence retrieval while using context more efficiently than shallow-probe and fixed-k comparators. Meaningful effects are preregistered as +0.02 recall, +0.03 answerable-only CER, or at least 10% fewer actual context tokens under a -0.02 recall noninferiority margin. Negative and null outcomes must be reported.

## Data partitions and exposure control

All {p['partitions']['design_exposed']['count']} previously inspectable IDs are frozen in `DESIGN_EXPOSED_IDS.json` and excluded. A fresh 225-example training-only sample uses nine evidence-bearing strata: three QASPER answerable length strata and six MuSiQue answerability/hop strata. Deterministic within-stratum hashing assigns 162 tuning and 63 selection examples before outcomes are computed. QASPER insufficient examples are excluded because they have no gold evidence spans; MuSiQue insufficient examples retain residual support and are analyzed separately.

QASPER test and all other final held-out records are forbidden. V2 IDs must contain `:train:`; final QASPER IDs contain `:test:`, so disjointness can be verified structurally without opening held-out records.

## Method

Each run uses exactly one declared representation: semantic, paragraph, hierarchical_v2, or sentence-window. There is no per-query router, RRF, or representation pair. Retrieval starts with a depth-3 probe, continues in batches of three, uses deterministic percentile-normalized relevance, and stops via calibrated no-gain patience plus token, round, candidate, and chunk safety caps. Novelty, similarity-gap, marginal-utility, and redundancy gates are absent from the v2 primary path. Deduplication occurs after retrieval. Final k remains emergent.

Token operating points are 800, 1,200, and 1,600 tokens, with 2,000 as a safety maximum only. All embeddings use the frozen local all-MiniLM-L6-v2 snapshot. No LLM may be called in retrieval, parameter fitting, or selection.

## Outcomes and comparisons

Primary retrieval outcomes are unique gold-span recall, answerable-only CER, and actual context tokens. Answer token F1 is deferred until paid authorization. Secondary outcomes include precision, insufficient-question residual-support recovery, MRR, nDCG with a chunk-granularity caveat, k, latency, and stop reasons.

Comparisons use identical queries, embeddings, and maximum context allowance. Reports must show actual consumption, recall-token Pareto frontiers, and matched actual-token points (mean tokens within 10%). Adaptive retrieval is not required to fill the safety budget.

## Statistics and selection

Paired bootstrap and paired randomization use 20,000 resamples with seed 20260905; binary outcomes use exact McNemar tests; declared primary comparisons receive Holm correction. Parameters are fit on tuning only. On selection, only the twelve tuning-selected representation/token candidates are eligible. Candidates within 0.01 of best recall are tied, then answerable CER is maximized, actual tokens and candidates are minimized, and the preregistered representation/token order breaks exact ties. Prior semantic screening performance provides no bonus.

The final interpretation must be one of: better on retrieval quality, better only on efficiency, Pareto competitive, or unsupported, using the exact rules in `PROTOCOL.json`.

## Integrity, resume, and paid gates

Every run records commands, configuration, seed, versions, hashes, failures, retries, and fingerprints. Resume is rejected on any protocol, configuration, embedding, input, sample, seed, or code mismatch. Outputs from distinct configurations are never mixed.

A paid smoke test is allowed only after retrieval artifacts, tests, exact frozen-manifest reproduction, zero-overlap audit, frozen answer prompt/model, and explicit user authorization with call and cost ceilings. It may use design-exposed training examples only. Final held-out evaluation requires a successful smoke, unchanged hashes, an exact cost/resume plan, and separate explicit authorization.
"""


def main() -> int:
    if PROTOCOL_JSON.exists():
        raise SystemExit(f"refusing to overwrite frozen protocol: {PROTOCOL_JSON}")
    exposed, sources = design_exposed_ids()
    payload = {"status": "FROZEN", "frozen_date": "2026-09-05", "count": len(exposed),
               "ids_sha256": hashlib.sha256(("\n".join(exposed) + "\n").encode()).hexdigest(),
               "source_artifacts": sources, "ids": exposed}
    proto = protocol(payload["ids_sha256"], len(exposed))
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "DESIGN_EXPOSED_IDS.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    PROTOCOL_JSON.write_text(json.dumps(proto, indent=2), encoding="utf-8")
    (OUT / "PROTOCOL.md").write_text(markdown(proto), encoding="utf-8")
    print(json.dumps({"protocol": proto["protocol_version"], "design_exposed_ids": len(exposed),
                      "source_artifacts": len(sources)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
