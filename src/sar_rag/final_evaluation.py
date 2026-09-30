"""Frozen answer-generation evaluation plumbing with strict cache semantics.

This module is transport-agnostic. The deterministic answerer supports free dry
runs; Groq is instantiated only by the paid CLI after an authorization gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from typing import Iterable, Protocol


ANSWERABILITY = {"ANSWERABLE", "PARTIALLY_ANSWERABLE", "INSUFFICIENT"}
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answerability": {"type": "string", "enum": sorted(ANSWERABILITY)},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "evidence_ids": {
                        "type": "array",
                        "items": {"type": "string", "pattern": "^E[1-9][0-9]*$"},
                    },
                },
                "required": ["text", "evidence_ids"],
                "additionalProperties": False,
            },
        },
        "missing_information": {"type": "string"},
    },
    "required": ["answerability", "claims", "missing_information"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = "You are a strictly evidence-grounded RAG answerer."
USER_TEMPLATE_VERSION = "final-evaluation-answer-v1"


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def object_sha256(value: object) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence_hash(evidence: list[dict]) -> str:
    normalized = [
        {"evidence_id": item["evidence_id"], "text": item["text"]}
        for item in evidence
    ]
    return object_sha256(normalized)


def build_messages(question: str, evidence: list[dict]) -> tuple[list[dict], str]:
    blocks = "\n".join(
        f"[{item['evidence_id']}] {item['text']}" for item in evidence
    ) or "No evidence passed frozen retrieval."
    user = (
        "Use only the supplied evidence. Preserve conditions, negation, scope, dates, "
        "and source versions. Return ANSWERABLE only when all requested information is "
        "supported, PARTIALLY_ANSWERABLE when only part is supported, and INSUFFICIENT "
        "when the evidence cannot answer the question. Every factual claim must cite one "
        "or more supplied evidence IDs. Never use outside knowledge.\n\n"
        f"QUESTION:\n{question}\n\nEVIDENCE:\n{blocks}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    prompt_spec = {
        "version": USER_TEMPLATE_VERSION,
        "system": SYSTEM_PROMPT,
        "schema": ANSWER_SCHEMA,
    }
    return messages, object_sha256(prompt_spec)


def request_fingerprint(request: dict, generation: dict) -> str:
    _, prompt_hash = build_messages(request["question"], request["evidence"])
    return object_sha256({
        "dataset": request["dataset"],
        "question_id": request["question_id"],
        "system": request["system"],
        "evidence_hash": evidence_hash(request["evidence"]),
        "prompt_hash": prompt_hash,
        "model": generation["model"],
        "decoding": generation["decoding"],
    })


def logical_key(request: dict) -> str:
    return "|".join((request["dataset"], request["question_id"], request["system"]))


def validate_request(request: dict) -> None:
    required = {"dataset", "question_id", "system", "question", "evidence"}
    missing = required - set(request)
    if missing:
        raise ValueError(f"request missing fields: {sorted(missing)}")
    ids = [item.get("evidence_id") for item in request["evidence"]]
    if ids != [f"E{i}" for i in range(1, len(ids) + 1)]:
        raise ValueError("evidence IDs must be contiguous E1..En in prompt order")
    if any(not str(item.get("text", "")).strip() for item in request["evidence"]):
        raise ValueError("evidence text must be non-empty")


def validate_answer(payload: dict, evidence_count: int) -> dict:
    if set(payload) != {"answerability", "claims", "missing_information"}:
        raise ValueError("answer response does not match the frozen schema")
    answerability = payload["answerability"]
    if answerability not in ANSWERABILITY:
        raise ValueError("invalid answerability value")
    allowed = {f"E{i}" for i in range(1, evidence_count + 1)}
    claims = payload["claims"]
    if not isinstance(claims, list) or not isinstance(payload["missing_information"], str):
        raise ValueError("invalid claims or missing_information type")
    for claim in claims:
        if set(claim) != {"text", "evidence_ids"}:
            raise ValueError("claim does not match frozen schema")
        ids = claim["evidence_ids"]
        if not str(claim["text"]).strip() or not ids or not set(ids) <= allowed:
            raise ValueError("claim is empty, uncited, or cites unknown evidence")
    if answerability != "INSUFFICIENT" and not claims:
        raise ValueError("answerable output must contain a cited claim")
    if answerability == "INSUFFICIENT" and claims:
        raise ValueError("insufficient output must not contain factual claims")
    if not evidence_count and answerability != "INSUFFICIENT":
        raise ValueError("empty evidence requires INSUFFICIENT")
    rendered = " ".join(
        f"{claim['text'].strip()} " + "".join(f"[{eid}]" for eid in claim["evidence_ids"])
        for claim in claims
    ).strip()
    if answerability == "INSUFFICIENT":
        rendered = payload["missing_information"].strip() or "The supplied evidence is insufficient."
    elif answerability == "PARTIALLY_ANSWERABLE" and payload["missing_information"].strip():
        rendered += " Missing information: " + payload["missing_information"].strip()
    return {**payload, "rendered_answer": rendered}


@dataclass(frozen=True)
class AnswerResponse:
    payload: dict
    input_tokens: int
    output_tokens: int
    latency_ms: float
    retries: int = 0


class Answerer(Protocol):
    def answer(self, request: dict, generation: dict) -> AnswerResponse: ...


class DeterministicFakeAnswerer:
    """Never imports or invokes a network client."""

    def __init__(self) -> None:
        self.calls = 0

    def answer(self, request: dict, generation: dict) -> AnswerResponse:
        self.calls += 1
        evidence = request["evidence"]
        if evidence:
            payload = {
                "answerability": "ANSWERABLE",
                "claims": [{"text": "Deterministic dry-run claim.", "evidence_ids": ["E1"]}],
                "missing_information": "",
            }
        else:
            payload = {
                "answerability": "INSUFFICIENT",
                "claims": [],
                "missing_information": "The supplied evidence is insufficient.",
            }
        input_tokens = 40 + len(request["question"].split()) + sum(
            len(item["text"].split()) for item in evidence
        )
        output_tokens = 12 if evidence else 8
        return AnswerResponse(payload, input_tokens, output_tokens, 1.0, 0)


class CheckpointRunner:
    def __init__(self, directory: Path, run_fingerprint: str, generation: dict,
                 pricing: dict):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.meta_path = directory / "CHECKPOINT_META.json"
        self.records_path = directory / "ANSWERS.jsonl"
        self.failures_path = directory / "FAILURES.jsonl"
        expected_meta = {
            "run_fingerprint": run_fingerprint,
            "generation_fingerprint": object_sha256(generation),
            "generation": generation,
        }
        if self.meta_path.exists():
            existing = json.loads(self.meta_path.read_text(encoding="utf-8"))
            if existing != expected_meta:
                raise ValueError("checkpoint configuration fingerprint mismatch")
        else:
            self.meta_path.write_text(
                json.dumps(expected_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        self.generation = generation
        self.pricing = pricing
        self.records: dict[str, dict] = {}
        if self.records_path.exists():
            for line in self.records_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    self.records[record["logical_key"]] = record

    def run(self, requests: Iterable[dict], answerer: Answerer) -> dict:
        new_calls = cache_hits = 0
        for request in requests:
            validate_request(request)
            key = logical_key(request)
            fingerprint = request_fingerprint(request, self.generation)
            existing = self.records.get(key)
            if existing:
                if existing["request_fingerprint"] != fingerprint:
                    raise ValueError(f"resume request fingerprint mismatch: {key}")
                cache_hits += 1
                continue
            started = perf_counter()
            try:
                response = answerer.answer(request, self.generation)
                validated = validate_answer(response.payload, len(request["evidence"]))
            except Exception as exc:
                failure = {
                    "logical_key": key,
                    "request_fingerprint": fingerprint,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                with self.failures_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(failure, sort_keys=True) + "\n")
                raise
            cost = (
                response.input_tokens * self.pricing["input_per_million"]
                + response.output_tokens * self.pricing["output_per_million"]
            ) / 1_000_000
            record = {
                "logical_key": key,
                "request_fingerprint": fingerprint,
                "dataset": request["dataset"],
                "question_id": request["question_id"],
                "system": request["system"],
                "evidence_hash": evidence_hash(request["evidence"]),
                "response": validated,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "total_tokens": response.input_tokens + response.output_tokens,
                "latency_ms": response.latency_ms,
                "runner_wall_ms": (perf_counter() - started) * 1000,
                "retries": response.retries,
                "estimated_cost_usd": cost,
            }
            with self.records_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            self.records[key] = record
            new_calls += 1
        return {
            "requested": new_calls + cache_hits,
            "new_calls": new_calls,
            "cache_hits": cache_hits,
            "checkpoint_records": len(self.records),
        }


def token_f1(prediction: str, references: list[str]) -> float:
    def normalize(text: str) -> list[str]:
        return "".join(ch.lower() if ch.isalnum() else " " for ch in text).split()
    pred = normalize(prediction)
    best = 0.0
    for reference in references:
        gold = normalize(reference)
        overlap = 0
        remaining = list(gold)
        for token in pred:
            if token in remaining:
                remaining.remove(token)
                overlap += 1
        if not pred or not gold:
            score = float(pred == gold)
        else:
            precision, recall = overlap / len(pred), overlap / len(gold)
            score = 2 * precision * recall / (precision + recall) if overlap else 0.0
        best = max(best, score)
    return best
