"""Prepare frozen smoke-test evidence for all systems without any LLM call."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sar_rag.adaptive_pipeline import SarRagSystem  # noqa: E402
from sar_rag.adaptive_single_view_v2 import AdaptiveSingleViewV2Config, AdaptiveSingleViewV2System  # noqa: E402
from sar_rag.benchmark import _documents  # noqa: E402
from sar_rag.calibration import RetrievalOnlyAnswerer  # noqa: E402
from sar_rag.config import SarRagConfig  # noqa: E402
from sar_rag.corpus import CorpusArtifacts, build_corpus  # noqa: E402
from sar_rag.dataset_adapters import iter_musique_full, iter_qasper  # noqa: E402
from sar_rag.encoding import SentenceTransformerEncoder  # noqa: E402
from sar_rag.final_evaluation import file_sha256, object_sha256  # noqa: E402

OUT = ROOT / "output" / "final_evaluation"
MANIFEST = OUT / "SMOKE_TEST_MANIFEST.json"
PROTOCOL = OUT / "FINAL_EVALUATION_PROTOCOL.json"
REQUESTS = OUT / "SMOKE_TEST_REQUESTS.jsonl"
GOLD = OUT / "SMOKE_TEST_GOLD_DESIGN_EXPOSED.jsonl"
REQUEST_MANIFEST = OUT / "SMOKE_TEST_REQUESTS_MANIFEST.json"
RECOMPUTE_VALIDATION = OUT / "SMOKE_TEST_REQUESTS_RECOMPUTATION_VALIDATION.json"


def write_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen smoke artifact: {path}")
    path.write_text(text, encoding="utf-8")


def pack(items, encoder, cap: int = 1600) -> tuple[list[dict], int]:
    packed, used = [], 0
    for item in items:
        chunk = item.chunk if hasattr(item, "chunk") else item
        tokens = encoder.token_count(chunk.evidence_text)
        if used + tokens > cap:
            continue
        packed.append({"evidence_id": f"E{len(packed) + 1}", "text": chunk.evidence_text})
        used += tokens
    return packed, used


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    smoke = json.loads(MANIFEST.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    wanted = {item["example_id"] for item in smoke["examples"]}
    raw = ROOT / "datasets" / "raw"
    rows = {}
    for row in iter_qasper(raw / "qasper/data/qasper-train-v0.3.json", "train"):
        if row["example_id"] in wanted:
            rows[row["example_id"]] = row
    for row in iter_musique_full(raw / "musique/data/musique_full_v1.0_train.jsonl", "train"):
        if row["example_id"] in wanted:
            rows[row["example_id"]] = row
    if set(rows) != wanted:
        raise RuntimeError(f"missing smoke rows: {sorted(wanted - set(rows))}")

    existing_requests = None
    if REQUESTS.exists():
        requests = [json.loads(line) for line in REQUESTS.read_text(encoding="utf-8").splitlines() if line.strip()]
        expected_keys = {(example_id, system) for example_id in wanted for system in smoke["systems"]}
        actual_keys = {(row["question_id"], row["system"]) for row in requests}
        if len(requests) != 48 or actual_keys != expected_keys:
            raise RuntimeError("existing frozen smoke requests do not match the frozen manifest")
        existing_encoder = SentenceTransformerEncoder(str(ROOT / "models/all-MiniLM-L6-v2"))
        packed_counts = [
            int(row["retrieval"]["packed_context_tokens"])
            if "retrieval" in row
            else sum(existing_encoder.token_count(item["text"]) for item in row["evidence"])
            for row in requests
        ]
        if any(count > 1600 for count in packed_counts):
            raise RuntimeError("existing frozen smoke request exceeds the common evidence cap")
        gold_rows = [{
            "dataset": rows[item["example_id"]]["dataset"],
            "question_id": item["example_id"],
            "gold_answerability": rows[item["example_id"]].get("gold_answerability"),
            "gold_answers": rows[item["example_id"]].get("gold_answers", []),
            "gold_evidence_spans": rows[item["example_id"]].get("gold_evidence_spans", []),
            "status": "DESIGN_EXPOSED_TRAINING_SMOKE_ONLY_NOT_RESEARCH_EVIDENCE",
        } for item in smoke["examples"]]
        if not GOLD.exists():
            write_new(GOLD, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in gold_rows))
        request_manifest = {
            "status": "FROZEN_RETRIEVAL_CONTEXTS_ZERO_LLM_CALLS",
            "question_count": len(smoke["examples"]), "request_count": len(requests),
            "systems": smoke["systems"], "common_packed_evidence_cap_tokens": 1600,
            "protocol_fingerprint": protocol["protocol_fingerprint"],
            "smoke_manifest_fingerprint": smoke["manifest_fingerprint"],
            "requests_sha256": file_sha256(REQUESTS), "gold_sha256": file_sha256(GOLD),
            "retrieval_only": True, "groq_client_imported_or_invoked": False,
            "existing_requests_preserved": True,
            "maximum_observed_packed_context_tokens": max(packed_counts),
        }
        request_manifest["manifest_fingerprint"] = object_sha256(request_manifest)
        if not REQUEST_MANIFEST.exists():
            write_new(REQUEST_MANIFEST, json.dumps(request_manifest, indent=2, sort_keys=True) + "\n")
        if not args.verify_existing:
            print(json.dumps(request_manifest, sort_keys=True))
            return
        existing_requests = requests

    v1_manifest = json.loads((ROOT / "output/calibration/frozen_manifest.json").read_text(encoding="utf-8"))
    v1_config = SarRagConfig(**v1_manifest["frozen_configuration"]["proposed"])
    v2_manifest = json.loads((ROOT / "output/adaptive_single_view_v2/recalibration_seed20260905/FROZEN_V2_MANIFEST.json").read_text(encoding="utf-8"))
    v2_config = AdaptiveSingleViewV2Config(**v2_manifest["selected_configuration"])
    encoder = SentenceTransformerEncoder(str(ROOT / "models/all-MiniLM-L6-v2"), revision=v2_config.embedding_revision)
    requests, gold_rows = [], []
    ordered_examples = smoke["examples"]
    for position, selected in enumerate(ordered_examples, 1):
        row = rows[selected["example_id"]]
        documents = _documents(row)
        artifacts = build_corpus(documents, encoder, ("sentence_window", "paragraph", "semantic", "hierarchical"))

        v1_system = SarRagSystem(artifacts, encoder, RetrievalOnlyAnswerer(), v1_config)
        v1_result = v1_system.ask(row["query"])
        v1_evidence, v1_packed_tokens = pack(v1_result.final_evidence, encoder)
        v1_done = next(event for event in reversed(v1_result.trace) if event.get("event") == "run_completed")

        v2_artifacts = CorpusArtifacts(artifacts.units_by_id, artifacts.chunks_by_id, {"paragraph": artifacts.indexes["paragraph"]})
        v2_system = AdaptiveSingleViewV2System(v2_artifacts, encoder, v2_config, v2_manifest["embedding"]["fingerprint"])
        v2_result = v2_system.retrieve(row["query"])
        v2_evidence, v2_packed_tokens = pack(v2_result.final_evidence, encoder)

        candidates = artifacts.indexes["sentence_window"].search(row["query"], 1, 10, "static")
        traditional_chunks = [artifacts.chunks_by_id[candidate.chunk_id] for candidate in candidates]
        traditional_evidence, traditional_tokens = pack(traditional_chunks, encoder)

        systems = {
            "traditional_k10": (traditional_evidence, {"native_context_tokens": traditional_tokens, "packed_context_tokens": traditional_tokens, "final_k": len(traditional_evidence), "stop_reason": "static_top_k"}),
            "sar_rag_v1": (v1_evidence, {"native_context_tokens": int(v1_done.get("final_context_tokens", 0)), "packed_context_tokens": v1_packed_tokens, "final_k": len(v1_evidence), "stop_reason": v1_result.stop_reason}),
            "adaptive_single_view_v2": (v2_evidence, {"native_context_tokens": v2_result.context_tokens, "packed_context_tokens": v2_packed_tokens, "final_k": len(v2_evidence), "stop_reason": v2_result.stop_reason}),
        }
        for system_name in smoke["systems"]:
            evidence, retrieval = systems[system_name]
            requests.append({
                "dataset": row["dataset"], "question_id": row["example_id"],
                "system": system_name, "question": row["query"], "evidence": evidence,
                "retrieval": retrieval, "smoke_position": position,
            })
        gold_rows.append({
            "dataset": row["dataset"], "question_id": row["example_id"],
            "gold_answerability": row.get("gold_answerability"),
            "gold_answers": row.get("gold_answers", []),
            "gold_evidence_spans": row.get("gold_evidence_spans", []),
            "status": "DESIGN_EXPOSED_TRAINING_SMOKE_ONLY_NOT_RESEARCH_EVIDENCE",
        })
        print(json.dumps({"prepared": position, "of": len(ordered_examples), "example_id": row["example_id"]}), flush=True)

    if len(requests) != 48 or len({(r["question_id"], r["system"]) for r in requests}) != 48:
        raise RuntimeError("smoke request count or logical-key uniqueness failure")
    if existing_requests is not None:
        def substantive(row: dict) -> dict:
            return {
                **{key: row[key] for key in ("dataset", "question_id", "system", "question")},
                "evidence": [
                    {"evidence_id": item["evidence_id"], "text": item["text"]}
                    for item in row["evidence"]
                ],
            }
        old = sorted((substantive(row) for row in existing_requests), key=lambda row: (row["question_id"], row["system"]))
        new = sorted((substantive(row) for row in requests), key=lambda row: (row["question_id"], row["system"]))
        validation = {
            "status": "RECOMPUTED_RETRIEVAL_ONLY_ZERO_LLM_CALLS",
            "substantive_identical": old == new,
            "request_count": len(new),
            "existing_requests_sha256": file_sha256(REQUESTS),
            "recomputed_substantive_sha256": object_sha256(new),
            "groq_calls": 0,
        }
        if not validation["substantive_identical"]:
            raise RuntimeError("existing smoke requests differ from fresh frozen retrieval recomputation")
        if not RECOMPUTE_VALIDATION.exists():
            write_new(RECOMPUTE_VALIDATION, json.dumps(validation, indent=2, sort_keys=True) + "\n")
        print(json.dumps(validation, sort_keys=True))
        return
    write_new(REQUESTS, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in requests))
    write_new(GOLD, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in gold_rows))
    request_manifest = {
        "status": "FROZEN_RETRIEVAL_CONTEXTS_ZERO_LLM_CALLS",
        "question_count": len(ordered_examples), "request_count": len(requests),
        "systems": smoke["systems"], "common_packed_evidence_cap_tokens": 1600,
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "smoke_manifest_fingerprint": smoke["manifest_fingerprint"],
        "requests_sha256": file_sha256(REQUESTS), "gold_sha256": file_sha256(GOLD),
        "retrieval_only": True, "groq_client_imported_or_invoked": False,
    }
    request_manifest["manifest_fingerprint"] = object_sha256(request_manifest)
    write_new(REQUEST_MANIFEST, json.dumps(request_manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(request_manifest, sort_keys=True))


if __name__ == "__main__":
    main()
