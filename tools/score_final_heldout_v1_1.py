"""Separately authorize and score sealed held-out v1.1 predictions (zero API calls)."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sar_rag.final_evaluation import file_sha256, token_f1  # noqa: E402
from sar_rag.smoke_recovery import load_jsonl  # noqa: E402


def normalize(text: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in text).split())


def covered(evidence: list[dict], gold: list[str]) -> set[int]:
    result: set[int] = set()
    normalized_gold = [normalize(span) for span in gold]
    for item in evidence:
        text = normalize(item["text"])
        result.update(index for index, span in enumerate(normalized_gold) if span and (span in text or (len(text.split()) >= 5 and text in span)))
    return result


def mean_defined(values: list[float]) -> float | None:
    clean = [value for value in values if not math.isnan(value)]
    return statistics.fmean(clean) if clean else None


def exact_match(prediction: str, references: list[str]) -> float:
    value = normalize(prediction)
    return float(any(value == normalize(reference) for reference in references))


def verify_authorization(path: Path | None, protocol: dict, manifest: dict,
                         answers: Path, attempts: Path, retrieval: Path) -> None:
    if path is None or not path.is_file():
        raise SystemExit("HELD-OUT SCORING BLOCKED: separately matching authorization is absent")
    actual = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "EXPLICIT_USER_AUTHORIZATION_RECORDED",
        "authorized_stage": "heldout_scoring_v1_1_only",
        "user_authorized": True,
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "run_manifest_fingerprint": manifest["manifest_fingerprint"],
        "answers_sha256": file_sha256(answers),
        "attempt_ledger_sha256": file_sha256(attempts),
        "retrieval_audit_sha256": file_sha256(retrieval),
        "judge_model_calls": 0,
    }
    if actual != expected:
        raise SystemExit("HELD-OUT SCORING BLOCKED: authorization content/fingerprint mismatch")


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - position) + values[upper] * (position - lower)


def paired_cluster_inference(rows: list[dict], metric: str, left: str, right: str,
                             seed: int = 20260926, replicates: int = 10000) -> dict:
    by_question: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_question[row["question_id"]][row["system"]] = row
    paired = []
    for question_id, systems in by_question.items():
        if left in systems and right in systems:
            a, b = systems[left][metric], systems[right][metric]
            if a is not None and b is not None and not math.isnan(float(a)) and not math.isnan(float(b)):
                paired.append((systems[left]["dependency_group"], float(a) - float(b)))
    clusters: dict[str, list[float]] = defaultdict(list)
    for cluster, difference in paired:
        clusters[cluster].append(difference)
    cluster_ids = sorted(clusters)
    if not cluster_ids:
        return {"paired_n": 0, "cluster_n": 0, "mean_difference": None, "ci95": None, "raw_p": None}
    rng = random.Random(f"{seed}|{metric}|{left}|{right}")
    observed = statistics.fmean(value for _, value in paired)
    boot = []
    for _ in range(replicates):
        selected = [rng.choice(cluster_ids) for _ in cluster_ids]
        values = [value for cluster in selected for value in clusters[cluster]]
        boot.append(statistics.fmean(values))
    randomized = []
    for _ in range(replicates):
        values = []
        for cluster in cluster_ids:
            sign = -1 if rng.random() < 0.5 else 1
            values.extend(sign * value for value in clusters[cluster])
        randomized.append(statistics.fmean(values))
    raw_p = (1 + sum(abs(value) >= abs(observed) for value in randomized)) / (replicates + 1)
    return {
        "paired_n": len(paired), "cluster_n": len(cluster_ids), "mean_difference": observed,
        "ci95": [percentile(boot, 0.025), percentile(boot, 0.975)], "raw_p": raw_p,
    }


def holm(items: list[tuple[str, float]]) -> dict[str, float]:
    ordered = sorted(items, key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * value))
        adjusted[name] = running
    return adjusted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--retrieval-audit", type=Path, required=True)
    parser.add_argument("--authorization-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--open-gold-and-score", action="store_true")
    args = parser.parse_args()
    if not args.open_gold_and_score:
        raise SystemExit("HELD-OUT SCORING BLOCKED: --open-gold-and-score was not provided")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    verify_authorization(args.authorization_file, protocol, manifest, args.answers, args.attempts, args.retrieval_audit)

    # Gold-bearing files are opened only after the exact scoring gate above.
    gold = {}
    for path in (ROOT / "datasets/normalized/qasper_test.jsonl", ROOT / "datasets/normalized/musique_full_validation.jsonl"):
        for row in load_jsonl(path):
            gold[row["example_id"]] = row
    answers = {row["logical_key"]: row for row in load_jsonl(args.answers)}
    attempts = load_jsonl(args.attempts)
    audits = {row["logical_key"]: row for row in load_jsonl(args.retrieval_audit)}
    rows = []
    for pair in manifest["pairs"]:
        key = pair["logical_key"]
        label = gold[pair["question_id"]]
        audit = audits[key]
        answer = answers.get(key)
        persistent_failure = answer is None
        response = answer["response"] if answer else {"answerability": "GENERATION_FAILURE", "claims": []}
        prediction = " ".join(claim["text"] for claim in response["claims"])
        gold_answerability = label["gold_answerability"]
        if persistent_failure:
            f1 = em = 0.0
        elif gold_answerability == "INSUFFICIENT":
            f1 = em = float(response["answerability"] == "INSUFFICIENT")
        else:
            f1 = token_f1(prediction, label["gold_answers"])
            em = exact_match(prediction, label["gold_answers"])
        gold_spans = list(dict.fromkeys(label.get("gold_evidence_spans", [])))
        native_covered = covered(audit["native_evidence"], gold_spans)
        supplied_covered = covered(audit["supplied_evidence"], gold_spans)
        has_gold = bool(gold_spans)
        rows.append({
            "logical_key": key, "dataset": pair["dataset"], "question_id": pair["question_id"],
            "system": pair["system"], "dependency_group": pair["dependency_group"],
            "persistent_generation_failure": persistent_failure,
            "answer_token_f1": f1, "answer_exact_match": em,
            "answerability_correct": float(response["answerability"] == gold_answerability),
            "correct_abstention": float(gold_answerability == "INSUFFICIENT" and response["answerability"] == "INSUFFICIENT"),
            "incorrect_abstention": float(gold_answerability != "INSUFFICIENT" and response["answerability"] == "INSUFFICIENT"),
            "false_answer_on_insufficient": float(gold_answerability == "INSUFFICIENT" and response["answerability"] in {"ANSWERABLE", "PARTIALLY_ANSWERABLE"}),
            "native_evidence_recall": len(native_covered) / len(gold_spans) if has_gold else float("nan"),
            "supplied_evidence_recall": len(supplied_covered) / len(gold_spans) if has_gold else float("nan"),
            "answerable_complete_evidence": float(len(supplied_covered) == len(gold_spans)) if has_gold and gold_answerability == "ANSWERABLE" else float("nan"),
            "insufficient_residual_support_recovery": len(supplied_covered) / len(gold_spans) if has_gold and gold_answerability == "INSUFFICIENT" else float("nan"),
            "native_context_tokens": audit["native_context_tokens"], "supplied_context_tokens": audit["supplied_context_tokens"],
            "final_k": audit["native_final_k"],
            "input_tokens": answer["input_tokens"] if answer else 0,
            "output_tokens": answer["output_tokens"] if answer else 0,
            "latency_ms": answer["latency_ms"] if answer else 0,
            "estimated_success_cost_usd": answer["estimated_cost_usd"] if answer else 0,
        })
    aggregates = {}
    metrics = ["answer_token_f1", "answer_exact_match", "answerability_correct", "correct_abstention", "incorrect_abstention", "false_answer_on_insufficient", "native_evidence_recall", "supplied_evidence_recall", "answerable_complete_evidence", "insufficient_residual_support_recovery", "native_context_tokens", "supplied_context_tokens", "final_k", "input_tokens", "output_tokens", "latency_ms", "estimated_success_cost_usd"]
    for dataset in sorted({row["dataset"] for row in rows}):
        for system in manifest["systems"]:
            subset = [row for row in rows if row["dataset"] == dataset and row["system"] == system]
            aggregates[f"{dataset}|{system}"] = {"n": len(subset), **{metric: mean_defined([float(row[metric]) for row in subset]) for metric in metrics}}
    comparisons = [("adaptive_single_view_v2", "sar_rag_v1"), ("sar_rag_v1", "traditional_k10"), ("adaptive_single_view_v2", "traditional_k10")]
    primary_metrics = ["answer_token_f1", "supplied_evidence_recall", "answerable_complete_evidence", "supplied_context_tokens"]
    inference = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        for metric in primary_metrics:
            family = []
            for left, right in comparisons:
                name = f"{left}_minus_{right}"
                result = paired_cluster_inference(dataset_rows, metric, left, right)
                inference[f"{dataset}|{metric}|{name}"] = result
                if result["raw_p"] is not None:
                    family.append((name, result["raw_p"]))
            adjusted = holm(family)
            for name, value in adjusted.items():
                inference[f"{dataset}|{metric}|{name}"]["holm_adjusted_p"] = value
    attempt_cost = sum(float(row["input_tokens"] or 0) * 0.15 / 1_000_000 + float(row["output_tokens"] or 0) * 0.60 / 1_000_000 for row in attempts)
    report = {
        "status": "SCORED_SEALED_HELDOUT_ZERO_JUDGE_CALLS",
        "protocol_fingerprint": protocol["protocol_fingerprint"], "manifest_fingerprint": manifest["manifest_fingerprint"],
        "valid_pair_count": len(rows), "aggregates": aggregates, "paired_inference": inference,
        "operational": {"attempts": len(attempts), "estimated_cost_all_attempts_usd": attempt_cost, "actual_invoiced_charge_usd": None},
        "limitations": protocol["interpretation_limits"], "judge_model_calls": 0,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "PER_PAIR_METRICS.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    (args.output / "RESULTS.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "pairs": len(rows), "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
