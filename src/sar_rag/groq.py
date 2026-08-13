from __future__ import annotations

import hashlib
import json
import os
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from time import perf_counter

from .models import Candidate, Chunk, FinalAnswerDecision, Finding, Judgment, Qualifier, QueryAnalysis, QueryFact, Requirement


class GroqClient:
    """Dependency-free Groq Chat Completions client with strict JSON support."""

    endpoint = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, model: str = "openai/gpt-oss-120b", api_key: str | None = None,
                 timeout_seconds: int = 90, max_retries: int = 8):
        self.model = model
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.usage_log: list[dict] = []
        self.attempt_log: list[dict] = []
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is required for Groq inference")

    def complete(self, messages: list[dict], schema_name: str | None = None,
                 schema: dict | None = None, max_tokens: int = 2048) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "top_p": 1,
            "max_completion_tokens": max_tokens,
            "reasoning_effort": "medium",
        }
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            }
        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "sar-rag-research/0.1",
            },
        )
        last_error = None
        for attempt in range(self.max_retries + 1):
            started = perf_counter()
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    parsed = json.loads(response.read().decode("utf-8"))
                    response_headers = response.headers
                usage = parsed.get("usage", {})
                self.usage_log.append({
                    "model": parsed.get("model", self.model),
                    "latency_seconds": perf_counter() - started,
                    "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                    "completion_tokens": int(usage.get("completion_tokens", 0)),
                    "total_tokens": int(usage.get("total_tokens", 0)),
                    "cached_tokens": int(usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)),
                    "rate_limit_remaining_requests": response_headers.get("x-ratelimit-remaining-requests"),
                    "rate_limit_remaining_tokens": response_headers.get("x-ratelimit-remaining-tokens"),
                })
                self.attempt_log.append({"status": "success", "attempt": attempt + 1,
                                         "latency_seconds": perf_counter() - started})
                return parsed["choices"][0]["message"].get("content") or ""
            except HTTPError as exc:
                last_error = exc
                body = exc.read().decode("utf-8", "replace")[:2000]
                self.attempt_log.append({"status": "http_error", "attempt": attempt + 1,
                                         "code": exc.code, "latency_seconds": perf_counter() - started})
                if exc.code == 429 and attempt < self.max_retries:
                    header_delay = exc.headers.get("retry-after")
                    match = re.search(r"try again in\s+([0-9.]+)\s*(ms|s)", body, re.IGNORECASE)
                    if header_delay:
                        delay = float(header_delay)
                    elif match:
                        delay = float(match.group(1)) / (1000 if match.group(2).lower() == "ms" else 1)
                    else:
                        delay = min(2 ** attempt, 60)
                    time.sleep(min(max(delay + 0.25, 1.0), 60.0))
                    continue
                raise RuntimeError(f"Groq HTTP {exc.code}: {body or exc.reason}") from exc
            except (URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                self.attempt_log.append({"status": type(exc).__name__, "attempt": attempt + 1,
                                         "latency_seconds": perf_counter() - started})
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 60))
        raise RuntimeError(f"Groq request failed after retries: {last_error}")


QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "requirements": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "text": {"type": "string"}, "critical": {"type": "boolean"},
                "entities": {"type": "array", "items": {"type": "string"}},
                "applicability_variables": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["text", "critical", "entities", "applicability_variables"],
            "additionalProperties": False,
        }},
        "query_facts": {"type": "array", "items": {"type": "string"}},
        "decomposition_confident": {"type": "boolean"},
    },
    "required": ["requirements", "query_facts", "decomposition_confident"],
    "additionalProperties": False,
}


class GroqRequirementExtractor:
    def __init__(self, client: GroqClient):
        self.client = client

    def analyze(self, query: str) -> QueryAnalysis:
        prompt = (
            "Extract the smallest set of independent evidence requirements needed to answer the query. "
            "Do not answer it or add facts. Separate facts explicitly stated in the query. "
            "Use one requirement for a simple question and at most eight.\n\nQUERY:\n" + query
        )
        raw = self.client.complete(
            [{"role": "system", "content": "You perform source-grounded query analysis."},
             {"role": "user", "content": prompt}],
            "query_analysis", QUERY_SCHEMA, 1200,
        )
        data = json.loads(raw)
        requirements = tuple(Requirement(
            f"R{i+1}", item["text"], item["critical"], tuple(item["entities"]),
            tuple(item["applicability_variables"]),
        ) for i, item in enumerate(data["requirements"][:8]))
        if not requirements:
            requirements = (Requirement("R1", query, True),)
        return QueryAnalysis(
            hashlib.sha256(query.encode()).hexdigest()[:24], query, requirements,
            tuple(QueryFact(f"QF{i+1}", text) for i, text in enumerate(data["query_facts"])),
            data["decomposition_confident"],
        )


def _judgment_schema(requirement_ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "findings": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "requirement_id": {"type": "string", "enum": requirement_ids},
                    "status": {"type": "string", "enum": ["SUPPORTED", "REFUTED", "NOT_APPLICABLE", "UNRESOLVED"]},
                    "claim": {"type": ["string", "null"]},
                    "cited_unit_ids": {"type": "array", "items": {"type": "string"}},
                    "applicability": {"type": "array", "items": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}, "values": {"type": "array", "items": {"type": "string"}}},
                        "required": ["name", "values"], "additionalProperties": False,
                    }},
                    "qualifiers": {"type": "array", "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"}, "text": {"type": "string"},
                            "cited_unit_ids": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["type", "text", "cited_unit_ids"], "additionalProperties": False,
                    }},
                },
                "required": ["requirement_id", "status", "claim", "cited_unit_ids", "applicability", "qualifiers"],
                "additionalProperties": False,
            }},
            "context_keys": {"type": "array", "items": {"type": "string"}},
            "contradiction_keys": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["findings", "context_keys", "contradiction_keys"],
        "additionalProperties": False,
    }


class GroqEvidenceJudge:
    def __init__(self, client: GroqClient):
        self.client = client

    def judge(self, query: QueryAnalysis, candidate: Candidate, chunk: Chunk) -> Judgment:
        requirements = "\n".join(f"{r.requirement_id}: {r.text}" for r in query.requirements)
        facts = "\n".join(f.text for f in query.query_facts) or "None"
        prompt = f"""Judge only supplied evidence; never use outside knowledge. Every non-UNRESOLVED
finding must cite allowed unit IDs. Do not transfer rules across entities, departments,
dates, jurisdictions, or versions. Absence is UNRESOLVED, never REFUTED.

QUERY: {query.original_query}
QUERY FACTS:\n{facts}
REQUIREMENTS:\n{requirements}
ALLOWED UNIT IDS: {', '.join(candidate.canonical_unit_ids)}
REPRESENTATION: {candidate.representation_id}
SOURCE DOCUMENT ID: {candidate.document_id}
SOURCE VERSION ID: {candidate.version_id}
SOURCE METADATA: {json.dumps(chunk.metadata, ensure_ascii=False, sort_keys=True)}
SOURCE TEXT:\n{chunk.evidence_text}"""
        data = json.loads(self.client.complete(
            [{"role": "system", "content": "You are a strict evidence and applicability judge."},
             {"role": "user", "content": prompt}],
            "evidence_judgment", _judgment_schema([r.requirement_id for r in query.requirements]), 2200,
        ))
        allowed = set(candidate.canonical_unit_ids)
        findings, valid = [], True
        for item in data["findings"]:
            cited = tuple(x for x in item["cited_unit_ids"] if x in allowed)
            if item["status"] != "UNRESOLVED" and not cited:
                valid = False
            qualifiers = tuple(Qualifier(
                q["type"], q["text"], tuple(x for x in q["cited_unit_ids"] if x in allowed)
            ) for q in item["qualifiers"])
            applicability = tuple((x["name"], tuple(x["values"])) for x in item["applicability"])
            findings.append(Finding(item["requirement_id"], item["status"], item["claim"], cited, applicability, qualifiers))
        return Judgment(candidate.candidate_id, tuple(findings), tuple(data["context_keys"]), tuple(data["contradiction_keys"]), valid=valid)


ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "answerability": {"type": "string", "enum": ["ANSWERABLE", "ANSWERABLE_WITH_CONFLICT", "PARTIALLY_ANSWERABLE", "INSUFFICIENT", "INSUFFICIENT_UNDER_BUDGET"]},
        "unresolved_requirements": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "answerability", "unresolved_requirements"],
    "additionalProperties": False,
}


class GroqAnswerGenerator:
    def __init__(self, client: GroqClient):
        self.client = client

    def generate(self, query: QueryAnalysis, evidence: list, unresolved: tuple[str, ...], answerability: str) -> str:
        blocks = [
            f"[E{i}] document={x.candidate.document_id} version={x.candidate.version_id} "
            f"metadata={json.dumps(x.chunk.metadata, ensure_ascii=False, sort_keys=True)} "
            f"units={','.join(x.candidate.canonical_unit_ids)}\n{x.chunk.evidence_text}"
            for i, x in enumerate(evidence, 1)
        ]
        prompt = f"""Answer only from supplied evidence. Preserve scope, conditions, exceptions,
negations, dates, and versions. Cite claims with [E#]. Do not fill unresolved gaps.
Required answerability: {answerability}
QUERY: {query.original_query}
UNRESOLVED IDS: {', '.join(unresolved) or 'None'}
EVIDENCE:\n{chr(10).join(blocks)}"""
        data = json.loads(self.client.complete(
            [{"role": "system", "content": "You write strictly source-grounded answers."},
             {"role": "user", "content": prompt}],
            "grounded_answer", ANSWER_SCHEMA, 1600,
        ))
        if data["answerability"] != answerability:
            raise RuntimeError("Groq answerability disagrees with deterministic pipeline state")
        if set(data["unresolved_requirements"]) != set(unresolved):
            raise RuntimeError("Groq unresolved-requirement list disagrees with deterministic pipeline state")
        cited = {int(value) for value in re.findall(r"\[E(\d+)\]", data["answer"])}
        if any(index < 1 or index > len(evidence) for index in cited):
            raise RuntimeError("Groq answer contains an invalid evidence citation")
        if evidence and answerability in {"ANSWERABLE", "ANSWERABLE_WITH_CONFLICT", "PARTIALLY_ANSWERABLE"} and not cited:
            raise RuntimeError("Groq answer contains no evidence citations")
        return data["answer"]


class GroqReformulator:
    def __init__(self, client: GroqClient):
        self.client = client

    def reformulate(self, query: QueryAnalysis, requirement: Requirement) -> str:
        raw = self.client.complete([
            {"role": "system", "content": "Return only a short retrieval query, never an answer."},
            {"role": "user", "content": f"Original: {query.original_query}\nUnresolved: {requirement.text}"},
        ], max_tokens=80)
        return " ".join(raw.strip().split())[:300]


ONE_CALL_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answerability": {
            "type": "string",
            "enum": ["ANSWERABLE", "PARTIALLY_ANSWERABLE", "INSUFFICIENT"],
        },
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


class GroqOneCallAnswerGenerator:
    """The only online LLM operation in the primary research pipeline."""

    def __init__(self, client: GroqClient):
        self.client = client

    def generate(self, query: str, evidence: list) -> FinalAnswerDecision:
        blocks = [
            f"[E{index}] document={item.candidate.document_id} "
            f"version={item.candidate.version_id} "
            f"units={','.join(item.candidate.canonical_unit_ids)}\n"
            f"{item.chunk.evidence_text}"
            for index, item in enumerate(evidence, 1)
        ]
        prompt = f"""Use only the supplied evidence to answer the query. Preserve conditions,
exceptions, negation, scope, dates and source versions. If all requested information is
supported, return ANSWERABLE. If only part is supported, return PARTIALLY_ANSWERABLE.
If the evidence cannot answer the query, return INSUFFICIENT. Put every supported factual
statement in claims and attach the exact supporting evidence_ids. Put unresolved facts in
missing_information. Never use outside knowledge. Do not write citation markers yourself;
the application renders them deterministically from evidence_ids.

QUERY: {query}
EVIDENCE:
{chr(10).join(blocks) if blocks else 'No evidence passed deterministic retrieval.'}"""
        data = json.loads(self.client.complete(
            [{"role": "system", "content": "You are a strictly evidence-grounded RAG answerer."},
             {"role": "user", "content": prompt}],
            "one_call_grounded_answer", ONE_CALL_ANSWER_SCHEMA, 1600,
        ))
        allowed = {f"E{index}" for index in range(1, len(evidence) + 1)}
        answerability = data["answerability"]
        claims, citations, invalid = [], [], False
        for claim in data["claims"]:
            ids = tuple(dict.fromkeys(claim["evidence_ids"]))
            if not claim["text"].strip() or not ids or not set(ids) <= allowed:
                invalid = True
                continue
            citations.extend(ids)
            claims.append(f"{claim['text'].strip()} " + "".join(f"[{value}]" for value in ids))
        if not evidence and answerability != "INSUFFICIENT":
            invalid = True
        if answerability != "INSUFFICIENT" and not claims:
            invalid = True
        if invalid:
            return FinalAnswerDecision(
                "The single model response did not provide valid claim-level evidence citations; "
                "an evidence-grounded answer cannot be returned.",
                "INSUFFICIENT",
                (),
            )
        if answerability == "INSUFFICIENT":
            answer = data["missing_information"].strip() or "The supplied evidence is insufficient."
            return FinalAnswerDecision(answer, "INSUFFICIENT", ())
        answer = " ".join(claims)
        if answerability == "PARTIALLY_ANSWERABLE" and data["missing_information"].strip():
            answer += " Missing information: " + data["missing_information"].strip()
        return FinalAnswerDecision(answer, answerability, tuple(dict.fromkeys(citations)))
