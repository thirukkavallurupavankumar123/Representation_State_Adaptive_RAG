"""Build and freeze held-out retrieval contexts without reading gold labels or calling an API."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from dataclasses import replace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sar_rag.adaptive_pipeline import SarRagSystem  # noqa: E402
from sar_rag.adaptive_single_view_v2 import AdaptiveSingleViewV2Config, AdaptiveSingleViewV2System  # noqa: E402
from sar_rag.benchmark import _documents  # noqa: E402
from sar_rag.calibration import RetrievalOnlyAnswerer  # noqa: E402
from sar_rag.config import SarRagConfig  # noqa: E402
from sar_rag.corpus import CorpusArtifacts, build_corpus  # noqa: E402
from sar_rag.encoding import SentenceTransformerEncoder  # noqa: E402
from sar_rag.index import ExactVectorIndex  # noqa: E402
from sar_rag.ingestion import parse_document  # noqa: E402
from sar_rag.representations import build_all  # noqa: E402
from sar_rag.final_evaluation import ANSWER_SCHEMA, file_sha256, logical_key, object_sha256  # noqa: E402
from sar_rag.generation_v1_1 import MAX_COMPLETION_TOKENS, build_chat_payload_v1_1  # noqa: E402
from sar_rag.heldout_v1_1 import heldout_request_fingerprint_v1_1  # noqa: E402
from sar_rag.smoke_recovery import load_jsonl  # noqa: E402


OUT = ROOT / "output/final_evaluation/heldout_v1_1"
PROTOCOL = OUT / "FINAL_EVALUATION_PROTOCOL_AMENDMENT_V1_1.json"
MANIFEST = OUT / "FROZEN_QUESTION_SYSTEM_MANIFEST.json"
QUESTIONS = OUT / "GENERATION_QUESTIONS.jsonl"
DOCUMENTS = OUT / "GENERATION_DOCUMENTS.jsonl"
REQUESTS = OUT / "REQUESTS.jsonl"
AUDIT = OUT / "RETRIEVAL_AND_PACKING_AUDIT.jsonl"
REQUEST_MANIFEST = OUT / "REQUEST_MANIFEST.json"
OFFLINE_CHECKPOINT = OUT / "OFFLINE_RETRIEVAL_GROUP_CHECKPOINT.jsonl"
V1 = ROOT / "output/calibration/frozen_manifest.json"
V2 = ROOT / "output/adaptive_single_view_v2/recalibration_seed20260905/FROZEN_V2_MANIFEST.json"

_WORKER: dict = {}


class MemoizingEncoder:
    """Reuse exact deterministic embeddings for repeated chunk/query text."""

    def __init__(self, base: SentenceTransformerEncoder):
        self.base = base
        self.dimension = base.dimension
        self.cache: dict[str, np.ndarray] = {}

    def encode(self, texts: list[str]) -> np.ndarray:
        missing = list(dict.fromkeys(text for text in texts if text not in self.cache))
        if missing:
            vectors = self.base.encode(missing)
            self.cache.update(zip(missing, vectors))
        return np.stack([self.cache[text] for text in texts]).astype(np.float32, copy=False) if texts else np.empty((0, self.dimension), dtype=np.float32)

    def preload(self, texts: list[str], batch_size: int = 64) -> None:
        """Encode missing texts in a deterministic fixed batch size."""
        missing = list(dict.fromkeys(text for text in texts if text not in self.cache))
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]
            vectors = self.base.encode(batch)
            self.cache.update(zip(batch, vectors))

    def token_count(self, text: str) -> int:
        return self.base.token_count(text)

    def prepare_embedding_text(self, text: str, max_tokens: int = 256):
        return self.base.prepare_embedding_text(text, max_tokens)


def chunk_of(item):
    return item.chunk if hasattr(item, "chunk") else item


def pack(items, encoder, cap: int = 1600) -> tuple[list[dict], list[dict], int]:
    supplied, exclusions, used = [], [], 0
    for rank, item in enumerate(items, 1):
        chunk = chunk_of(item)
        tokens = encoder.token_count(chunk.evidence_text)
        entry = {
            "native_rank": rank,
            "chunk_id": chunk.chunk_id,
            "representation": chunk.representation_id,
            "text_sha256": object_sha256(chunk.evidence_text),
            "tokens": tokens,
        }
        if used + tokens > cap:
            exclusions.append({**entry, "reason": "complete_chunk_would_exceed_common_cap"})
            continue
        supplied.append({"evidence_id": f"E{len(supplied) + 1}", "text": chunk.evidence_text, **entry})
        used += tokens
    return supplied, exclusions, used


def request_evidence(supplied: list[dict]) -> list[dict]:
    return [{"evidence_id": row["evidence_id"], "text": row["text"]} for row in supplied]


def initialize_worker() -> None:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    v1_manifest = json.loads(V1.read_text(encoding="utf-8"))
    v2_manifest = json.loads(V2.read_text(encoding="utf-8"))
    v2_config = AdaptiveSingleViewV2Config(**v2_manifest["selected_configuration"])
    _WORKER.update({
        "v1_config": SarRagConfig(**v1_manifest["frozen_configuration"]["proposed"]),
        "v2_manifest": v2_manifest,
        "v2_config": v2_config,
        "encoder": MemoizingEncoder(SentenceTransformerEncoder(str(ROOT / "models/all-MiniLM-L6-v2"), revision=v2_config.embedding_revision)),
    })


def prepare_group_from_artifacts(document_key: str, questions: list[dict], artifacts: CorpusArtifacts) -> tuple[str, list[tuple[dict, dict]]]:
    encoder = _WORKER["encoder"]
    output: list[tuple[dict, dict]] = []
    for question in questions:
        query = question["query"]
        v1_result = SarRagSystem(artifacts, encoder, RetrievalOnlyAnswerer(), _WORKER["v1_config"]).ask(query)
        v1_supplied, v1_excluded, v1_tokens = pack(v1_result.final_evidence, encoder)
        v1_done = next(event for event in reversed(v1_result.trace) if event.get("event") == "run_completed")
        v2_artifacts = CorpusArtifacts(artifacts.units_by_id, artifacts.chunks_by_id, {"paragraph": artifacts.indexes["paragraph"]})
        v2_result = AdaptiveSingleViewV2System(v2_artifacts, encoder, _WORKER["v2_config"], _WORKER["v2_manifest"]["embedding"]["fingerprint"]).retrieve(query)
        v2_supplied, v2_excluded, v2_tokens = pack(v2_result.final_evidence, encoder)
        traditional_candidates = artifacts.indexes["sentence_window"].search(query, 1, 10, "static")
        traditional_native = [artifacts.chunks_by_id[item.chunk_id] for item in traditional_candidates]
        traditional_supplied, traditional_excluded, traditional_tokens = pack(traditional_native, encoder)
        systems = {
            "traditional_k10": (traditional_native, traditional_supplied, traditional_excluded, traditional_tokens, len(traditional_native), "static_top_k", traditional_tokens),
            "sar_rag_v1": (v1_result.final_evidence, v1_supplied, v1_excluded, v1_tokens, v1_result.final_k, v1_result.stop_reason, int(v1_done.get("final_context_tokens", 0))),
            "adaptive_single_view_v2": (v2_result.final_evidence, v2_supplied, v2_excluded, v2_tokens, v2_result.final_k, v2_result.stop_reason, v2_result.context_tokens),
        }
        for system in ("traditional_k10", "sar_rag_v1", "adaptive_single_view_v2"):
            native, supplied, exclusions, supplied_tokens, final_k, stop_reason, native_tokens = systems[system]
            request = {"dataset": question["dataset"], "question_id": question["question_id"], "system": system, "question": query, "evidence": request_evidence(supplied)}
            audit = {
                "logical_key": logical_key(request), "dataset": question["dataset"], "question_id": question["question_id"],
                "system": system, "dependency_group": question["dependency_group"],
                "configuration_fingerprint": (
                    object_sha256(_WORKER["v1_config"].__dict__) if system == "sar_rag_v1"
                    else _WORKER["v2_config"].fingerprint if system == "adaptive_single_view_v2"
                    else object_sha256({"representation": "sentence_window", "fixed_k": 10})
                ),
                "native_final_k": final_k, "native_context_tokens": native_tokens,
                "native_evidence": [{"native_rank": rank, "chunk_id": chunk_of(item).chunk_id, "representation": chunk_of(item).representation_id, "text": chunk_of(item).evidence_text} for rank, item in enumerate(native, 1)],
                "supplied_evidence": supplied, "supplied_context_tokens": supplied_tokens,
                "packing_exclusions": exclusions, "packing_truncated_chunk_count": 0,
                "packing_skipped_complete_chunk_count": len(exclusions), "stop_reason": stop_reason,
            }
            output.append((request, audit))
    return document_key, output


def prepare_group(task: tuple[str, list[dict], list[dict]]) -> tuple[str, list[tuple[dict, dict]]]:
    """Reference single-group path retained for exact reproducibility checks."""
    document_key, document_rows, questions = task
    artifacts = build_corpus(_documents({"documents": document_rows}), _WORKER["encoder"], ("sentence_window", "paragraph", "semantic", "hierarchical"))
    return prepare_group_from_artifacts(document_key, questions, artifacts)


def prepare_batch(tasks: list[tuple[str, list[dict], list[dict]]]) -> list[tuple[str, list[tuple[dict, dict]]]]:
    """Batch embedding inference while keeping each document's exact retrieval isolated."""
    encoder = _WORKER["encoder"]
    parsed: dict[str, list[tuple[object, list[object]]]] = {}
    unit_texts: list[str] = []
    for document_key, document_rows, _ in tasks:
        values = []
        for document in _documents({"documents": document_rows}):
            units = parse_document(document)
            values.append((document, units))
            unit_texts.extend(unit.raw_text for unit in units if unit.unit_type != "heading")
        parsed[document_key] = values
    encoder.preload(unit_texts)

    prepared: dict[str, tuple[dict, dict[str, list[object]]]] = {}
    embedding_texts: list[str] = []
    for document_key, _, _ in tasks:
        all_units: dict = {}
        by_rep: dict[str, list[object]] = {rep: [] for rep in ("sentence_window", "paragraph", "semantic", "hierarchical")}
        for document, units in parsed[document_key]:
            all_units.update({unit.unit_id: unit for unit in units})
            representations = build_all(document, units, encoder, tuple(by_rep))
            for rep in by_rep:
                for chunk in representations[rep]:
                    embedding_text, original_embedding_tokens, was_truncated = encoder.prepare_embedding_text(chunk.embedding_text, 256)
                    prepared_chunk = replace(chunk, metadata={
                        **chunk.metadata,
                        "document_title": document.title,
                        "effective_date": document.effective_date,
                        "authority_level": document.authority_level,
                        "supersedes_version_id": document.supersedes_version_id,
                        "original_embedding_tokens": original_embedding_tokens,
                        "embedding_was_truncated": was_truncated,
                    }, embedding_text=embedding_text)
                    by_rep[rep].append(prepared_chunk)
                    embedding_texts.append(embedding_text)
        prepared[document_key] = (all_units, by_rep)
    encoder.preload(embedding_texts)
    encoder.preload([question["query"] for _, _, questions in tasks for question in questions])

    results = []
    for document_key, _, questions in tasks:
        all_units, by_rep = prepared[document_key]
        chunks = {chunk.chunk_id: chunk for values in by_rep.values() for chunk in values}
        indexes = {rep: ExactVectorIndex(rep, values, encoder) for rep, values in by_rep.items()}
        results.append(prepare_group_from_artifacts(document_key, questions, CorpusArtifacts(all_units, chunks, indexes)))
    return results


def main() -> int:
    for path in (REQUESTS, AUDIT, REQUEST_MANIFEST):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite frozen request artifact: {path}")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    questions = load_jsonl(QUESTIONS)
    documents = {row["document_key"]: row["documents"] for row in load_jsonl(DOCUMENTS)}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/gpt-oss-120b-tokenizer", local_files_only=True)
    serialized_schema = json.dumps(ANSWER_SCHEMA, separators=(",", ":"))
    expected_pairs = {row["logical_key"] for row in manifest["pairs"]}
    request_rows, audit_rows, request_pairs = [], [], []
    grouped: dict[str, list[dict]] = {}
    for question in questions:
        grouped.setdefault(question["document_key"], []).append(question)
    completed_groups: dict[str, list[tuple[dict, dict]]] = {}
    if OFFLINE_CHECKPOINT.exists():
        for line in OFFLINE_CHECKPOINT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                saved = json.loads(line)
                completed_groups[saved["document_key"]] = [(item["request"], item["audit"]) for item in saved["items"]]
    tasks = [(key, documents[key], sorted(values, key=lambda row: row["question_id"])) for key, values in sorted(grouped.items()) if key not in completed_groups]
    from concurrent.futures import ProcessPoolExecutor
    workers = min(6, max(1, (os.cpu_count() or 2) - 1))
    completed_questions = sum(len(items) // 3 for items in completed_groups.values())
    for items in completed_groups.values():
        for request, audit in items:
            key = logical_key(request)
            request_rows.append(request)
            audit_rows.append(audit)
            payload = build_chat_payload_v1_1(request["question"], request["evidence"])
            preflight = len(tokenizer.apply_chat_template(payload["messages"], tokenize=True, add_generation_prompt=True)) + len(tokenizer.encode(serialized_schema, add_special_tokens=False))
            conservative = preflight + 128
            if conservative > 2800:
                raise RuntimeError(f"predeclared 2,800-token input ceiling exceeded: {key}={conservative}")
            request_pairs.append({"logical_key": key, "heldout_request_fingerprint_v1_1": heldout_request_fingerprint_v1_1(request), "preflight_input_tokens": preflight, "conservative_billed_input_tokens": conservative})
    # Remaining QASPER papers are individually large enough to fill embedding
    # batches; one paper per task prevents multi-paper memory pressure.
    task_batch_size = 1
    task_batches = [tasks[start:start + task_batch_size] for start in range(0, len(tasks), task_batch_size)]
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker) as executor:
        for batch_result in executor.map(prepare_batch, task_batches, chunksize=1):
          for document_key, group_result in batch_result:
            with OFFLINE_CHECKPOINT.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"document_key": document_key, "items": [{"request": request, "audit": audit} for request, audit in group_result]}, ensure_ascii=False, sort_keys=True) + "\n")
            for request, audit in group_result:
                key = logical_key(request)
                request_rows.append(request)
                audit_rows.append(audit)
                payload = build_chat_payload_v1_1(request["question"], request["evidence"])
                preflight = len(tokenizer.apply_chat_template(payload["messages"], tokenize=True, add_generation_prompt=True)) + len(tokenizer.encode(serialized_schema, add_special_tokens=False))
                conservative = preflight + 128
                if conservative > 2800:
                    raise RuntimeError(f"predeclared 2,800-token input ceiling exceeded: {key}={conservative}")
                request_pairs.append({
                    "logical_key": key, "heldout_request_fingerprint_v1_1": heldout_request_fingerprint_v1_1(request),
                    "preflight_input_tokens": preflight, "conservative_billed_input_tokens": conservative,
                })
            completed_questions += len(group_result) // 3
            if completed_questions % 25 < len(group_result) // 3 or completed_questions == len(questions):
                print(json.dumps({"prepared_questions": completed_questions, "of": len(questions), "workers": workers, "embedding_batch_size": 64, "document_batch_size": task_batch_size}), flush=True)
    order = {row["logical_key"]: index for index, row in enumerate(manifest["pairs"])}
    request_rows.sort(key=lambda row: order[logical_key(row)])
    audit_rows.sort(key=lambda row: order[row["logical_key"]])
    request_pairs.sort(key=lambda row: order[row["logical_key"]])
    if {logical_key(row) for row in request_rows} != expected_pairs or len(request_rows) != manifest["pair_count"]:
        raise RuntimeError("prepared requests differ from the frozen pair manifest")
    REQUESTS.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in request_rows), encoding="utf-8")
    AUDIT.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in audit_rows), encoding="utf-8")
    request_manifest = {
        "status": "FROZEN_OFFLINE_RETRIEVAL_REQUESTS_ZERO_API_CALLS",
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "run_manifest_fingerprint": manifest["manifest_fingerprint"],
        "pair_count": len(request_rows), "requests_sha256": file_sha256(REQUESTS),
        "retrieval_audit_sha256": file_sha256(AUDIT),
        "maximum_preflight_input_tokens": max(row["preflight_input_tokens"] for row in request_pairs),
        "maximum_conservative_billed_input_tokens": max(row["conservative_billed_input_tokens"] for row in request_pairs),
        "conservative_cost_ceiling_compatible": True,
        "gold_labels_accessed": False, "api_calls": 0, "pairs": request_pairs,
    }
    request_manifest["request_manifest_fingerprint"] = object_sha256(request_manifest)
    REQUEST_MANIFEST.write_text(json.dumps(request_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in request_manifest.items() if key != "pairs"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
