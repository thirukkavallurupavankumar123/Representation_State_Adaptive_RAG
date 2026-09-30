"""Frozen smoke-generation v1.1 request construction.

This module is deliberately separate from the original v1 plumbing. It changes
only the prompt version and completion-token limit while retaining the original
model, schema, decoding choices, evidence, and semantic validator.
"""

from __future__ import annotations

from copy import deepcopy

from .final_evaluation import ANSWER_SCHEMA, evidence_hash, object_sha256


PROTOCOL_VERSION = "sar-rag-smoke-generation-1.1.0"
USER_TEMPLATE_VERSION = "final-evaluation-answer-v1.1"
SYSTEM_PROMPT = "You are a strictly evidence-grounded RAG answerer."
MODEL = "openai/gpt-oss-120b"
TEMPERATURE = 0
TOP_P = 1
REASONING_EFFORT = "medium"
MAX_COMPLETION_TOKENS = 1600
SCHEMA_NAME = "frozen_final_answer_v1_1"
STRICT = True


def build_messages_v1_1(question: str, evidence: list[dict]) -> tuple[list[dict], str]:
    blocks = "\n".join(
        f"[{item['evidence_id']}] {item['text']}" for item in evidence
    ) or "No evidence passed frozen retrieval."
    user = (
        "Use only the supplied evidence. Preserve conditions, negation, scope, dates, "
        "and source versions. Return ANSWERABLE only when all requested information is "
        "supported, PARTIALLY_ANSWERABLE when only part is supported, and INSUFFICIENT "
        "when the evidence cannot answer the question. Every factual claim must cite one "
        "or more supplied evidence IDs. Never use outside knowledge.\n\n"
        "Return exactly the three required top-level fields: answerability, claims, and "
        "missing_information. Always emit missing_information; use an empty string when "
        "nothing is missing. For INSUFFICIENT, return claims as an empty array and explain "
        "the gap in missing_information. For ANSWERABLE or PARTIALLY_ANSWERABLE, every "
        "claim must have non-empty text and at least one valid supplied evidence ID.\n\n"
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


def build_chat_payload_v1_1(question: str, evidence: list[dict]) -> dict:
    messages, _ = build_messages_v1_1(question, evidence)
    return {
        "model": MODEL,
        "messages": messages,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "reasoning_effort": REASONING_EFFORT,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": SCHEMA_NAME,
                "strict": STRICT,
                "schema": deepcopy(ANSWER_SCHEMA),
            },
        },
    }


def generation_config_v1_1() -> dict:
    return {
        "model": MODEL,
        "decoding": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "reasoning_effort": REASONING_EFFORT,
            "max_output_tokens": MAX_COMPLETION_TOKENS,
            "response_format": "strict JSON schema",
        },
        "prompt_version": USER_TEMPLATE_VERSION,
        "schema_name": SCHEMA_NAME,
        "strict": STRICT,
    }


def request_fingerprint_v1_1(request: dict) -> str:
    _, prompt_hash = build_messages_v1_1(request["question"], request["evidence"])
    return object_sha256({
        "dataset": request["dataset"],
        "question_id": request["question_id"],
        "system": request["system"],
        "evidence_hash": evidence_hash(request["evidence"]),
        "prompt_hash": prompt_hash,
        "generation": generation_config_v1_1(),
    })

