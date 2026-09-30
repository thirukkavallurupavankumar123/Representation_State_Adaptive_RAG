from __future__ import annotations

from collections import Counter
import math
import re
import string

from .models import PipelineResult


def normalize_answer(text: str) -> str:
    text = re.sub(r"\[E\d+\]", " ", text, flags=re.IGNORECASE)
    text = text.casefold()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold_answers: tuple[str, ...]) -> float:
    normalized = normalize_answer(prediction)
    return float(any(normalized == normalize_answer(gold) for gold in gold_answers))


def token_f1(prediction: str, gold_answers: tuple[str, ...]) -> float:
    return token_prf(prediction, gold_answers)[2]


def token_prf(prediction: str, gold_answers: tuple[str, ...]) -> tuple[float, float, float]:
    prediction_tokens = normalize_answer(prediction).split()
    best = (0.0, 0.0, 0.0)
    for gold in gold_answers:
        gold_tokens = normalize_answer(gold).split()
        common = sum((Counter(prediction_tokens) & Counter(gold_tokens)).values())
        if not prediction_tokens or not gold_tokens:
            score = float(prediction_tokens == gold_tokens)
        elif not common:
            score = 0.0
        else:
            precision = common / len(prediction_tokens)
            recall = common / len(gold_tokens)
            score = 2 * precision * recall / (precision + recall)
        candidate = (precision, recall, score) if prediction_tokens and gold_tokens and common else (
            score, score, score
        )
        if candidate[2] > best[2]:
            best = candidate
    return best


def evidence_span_recall(result: PipelineResult, gold_spans: tuple[str, ...]) -> float:
    """Exact normalized substring recall; avoids invented dataset-specific unit mappings.

    Recall is undefined without a gold-evidence denominator.  Returning 1.0 for
    examples the benchmark declares to have no relevant evidence would award a
    perfect retrieval score for retrieving nothing relevant and would inflate any
    aggregate that mixes answerable and unanswerable questions.
    """
    if not gold_spans:
        return float("nan")
    evidence = normalize_answer("\n".join(item.chunk.evidence_text for item in result.final_evidence))
    found = sum(normalize_answer(span) in evidence for span in gold_spans)
    return found / len(gold_spans)


def evaluate_result(result: PipelineResult, gold_answers: tuple[str, ...],
                    gold_spans: tuple[str, ...], gold_answerability: str | None = None) -> dict[str, float]:
    metrics = {
        "evidence_span_recall": evidence_span_recall(result, gold_spans),
        "final_k": float(result.final_k),
        "final_context_words": float(sum(item.chunk.word_count for item in result.final_evidence)),
        "selected_representation_count": float(len(result.selected_representations)),
    }
    # Unanswerable examples have no gold answer string. Scoring them as answer EM/F1=0
    # would incorrectly punish a correct insufficiency decision.
    if gold_answers:
        metrics["answer_exact_match"] = exact_match(result.answer, gold_answers)
        metrics["answer_token_f1"] = token_f1(result.answer, gold_answers)
    if gold_answerability is not None:
        metrics["answerability_accuracy"] = float(result.answerability == gold_answerability)
    completed = next((event for event in reversed(result.trace) if event.get("event") == "run_completed"), {})
    metrics["candidates_examined"] = float(completed.get("candidates_examined", 0))
    return metrics


def macro_average(rows: list[dict[str, float]]) -> dict[str, float]:
    """Macro-average over defined values only.

    Metrics that are undefined for an example (for example retrieval recall on a
    question with no gold evidence) are recorded as NaN and must be excluded from
    the denominator rather than propagated into the aggregate.
    """
    keys = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    output = {}
    for key in keys:
        defined = [
            float(row[key]) for row in rows
            if key in row and not math.isnan(float(row[key]))
        ]
        output[key] = sum(defined) / len(defined) if defined else float("nan")
    return output
