"""Exercise the frozen held-out runner with a deterministic fake answerer."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sar_rag.final_evaluation import file_sha256, object_sha256  # noqa: E402
from sar_rag.heldout_v1_1 import HeldoutV11Runner, heldout_request_fingerprint_v1_1  # noqa: E402
from sar_rag.smoke_recovery import AttemptOutcome, load_jsonl  # noqa: E402


OUT = ROOT / "output/final_evaluation/heldout_v1_1"
PROTOCOL = OUT / "FINAL_EVALUATION_PROTOCOL_AMENDMENT_V1_1.json"
MANIFEST = OUT / "FROZEN_QUESTION_SYSTEM_MANIFEST.json"
QUESTIONS = OUT / "GENERATION_QUESTIONS.jsonl"
DOCUMENTS = OUT / "GENERATION_DOCUMENTS.jsonl"
REQUEST_MANIFEST = OUT / "DRY_RUN_REQUEST_MANIFEST.json"
CHECKPOINT = OUT / "dry_run_checkpoint"
REPORT = OUT / "DRY_RUN_VALIDATION.json"


class FakeAnswerer:
    def __init__(self) -> None:
        self.calls = 0

    def attempt(self, request: dict) -> AttemptOutcome:
        self.calls += 1
        payload = {
            "answerability": "ANSWERABLE",
            "claims": [{"text": "Deterministic dry-run claim.", "evidence_ids": ["E1"]}],
            "missing_information": "",
        }
        return AttemptOutcome(status="success", payload=payload, raw_response=json.dumps(payload), input_tokens=100, output_tokens=20, latency_ms=1.0)


def main() -> int:
    if REPORT.exists() or REQUEST_MANIFEST.exists() or CHECKPOINT.exists():
        raise FileExistsError("refusing to overwrite held-out dry-run artifacts")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    question_rows = load_jsonl(QUESTIONS)
    by_id = {row["question_id"]: row for row in question_rows}
    requests = []
    for pair in manifest["pairs"]:
        question = by_id[pair["question_id"]]
        requests.append({
            "dataset": pair["dataset"],
            "question_id": pair["question_id"],
            "system": pair["system"],
            "question": question["query"],
            "evidence": [{"evidence_id": "E1", "text": "Synthetic evidence used only to test local plumbing."}],
        })
    request_manifest = {
        "status": "SYNTHETIC_LOCAL_DRY_RUN_ONLY",
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "run_manifest_fingerprint": manifest["manifest_fingerprint"],
        "pair_count": len(requests),
        "pairs": [{
            "logical_key": pair["logical_key"],
            "heldout_request_fingerprint_v1_1": heldout_request_fingerprint_v1_1(request),
            "conservative_billed_input_tokens": 2800,
        } for pair, request in zip(manifest["pairs"], requests)],
    }
    request_manifest["request_manifest_fingerprint"] = object_sha256(request_manifest)
    REQUEST_MANIFEST.write_text(json.dumps(request_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    temporary_checkpoint = Path(tempfile.mkdtemp(prefix="sar-rag-heldout-dry-run-"))
    try:
        fake = FakeAnswerer()
        runner = HeldoutV11Runner(
            temporary_checkpoint, protocol, manifest, request_manifest,
            protocol_path=PROTOCOL, manifest_path=MANIFEST, request_manifest_path=REQUEST_MANIFEST,
        )
        first = runner.run(requests, fake)
        resume_fake = FakeAnswerer()
        resume = HeldoutV11Runner(
            temporary_checkpoint, protocol, manifest, request_manifest,
            protocol_path=PROTOCOL, manifest_path=MANIFEST, request_manifest_path=REQUEST_MANIFEST,
        ).run(requests, resume_fake)
        shutil.copytree(temporary_checkpoint, CHECKPOINT)
    finally:
        shutil.rmtree(temporary_checkpoint, ignore_errors=True)

    forbidden = {"gold_answers", "gold_answerability", "gold_evidence_spans"}
    generation_question_keys = set().union(*(row.keys() for row in question_rows))
    generation_document_keys: set[str] = set()
    with DOCUMENTS.open(encoding="utf-8") as stream:
        for line in stream:
            generation_document_keys.update(json.loads(line).keys())
    answers = load_jsonl(CHECKPOINT / "ANSWERS.jsonl")
    attempts = load_jsonl(CHECKPOINT / "API_ATTEMPTS.jsonl")
    report = {
        "status": "PASSED_DETERMINISTIC_FAKE_ANSWERER_DRY_RUN",
        "api_calls": 0,
        "fake_calls": fake.calls,
        "pair_count": manifest["pair_count"],
        "first_run": first,
        "resume_run": resume,
        "resume_fake_calls": resume_fake.calls,
        "resume_zero_duplicate_calls": resume_fake.calls == 0 and resume["new_attempts"] == 0,
        "answer_records": len(answers),
        "attempt_records": len(attempts),
        "all_first_attempt_valid": all(row["attempt_for_pair"] == 1 and row["first_attempt_valid"] for row in answers),
        "gold_fields_absent_from_generation_questions": not (generation_question_keys & forbidden),
        "gold_fields_absent_from_generation_documents": not (generation_document_keys & forbidden),
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "request_manifest_fingerprint": request_manifest["request_manifest_fingerprint"],
        "artifact_sha256": {
            "checkpoint_meta": file_sha256(CHECKPOINT / "CHECKPOINT_META.json"),
            "answers": file_sha256(CHECKPOINT / "ANSWERS.jsonl"),
            "attempts": file_sha256(CHECKPOINT / "API_ATTEMPTS.jsonl"),
            "dry_run_request_manifest": file_sha256(REQUEST_MANIFEST),
        },
    }
    if not all((
        report["resume_zero_duplicate_calls"], report["all_first_attempt_valid"],
        report["gold_fields_absent_from_generation_questions"],
        report["gold_fields_absent_from_generation_documents"],
        len(answers) == manifest["pair_count"], len(attempts) == manifest["pair_count"],
    )):
        raise RuntimeError(f"held-out dry run failed: {report}")
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
