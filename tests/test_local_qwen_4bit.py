import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sar_rag import local_qwen_4bit as lq  # noqa: E402
from sar_rag.generation_v1_1 import build_messages_v1_1, request_fingerprint_v1_1  # noqa: E402


EVIDENCE = [{"evidence_id": "E1", "text": "Water freezes at 0 C."}, {"evidence_id": "E2", "text": "Other."}]


class SeedTests(unittest.TestCase):
    def test_seed_is_stable_and_order_independent(self):
        first = lq.question_seed("qasper", "qasper:test:abc")
        self.assertEqual(first, lq.question_seed("qasper", "qasper:test:abc"))
        self.assertNotEqual(first, lq.question_seed("qasper", "qasper:test:abd"))
        self.assertTrue(0 <= first < 2**32)
        # Known value pins the explicit SHA-256 rule (no Python hash()).
        self.assertEqual(lq.question_seed("d", "q", 1), int.from_bytes(__import__("hashlib").sha256(b"1|d|q").digest()[:4], "big"))

    def test_system_order_is_a_permutation(self):
        order = lq.system_order("musique_full", "x", 20260930)
        self.assertEqual(sorted(order), sorted(lq.SYSTEMS))
        self.assertEqual(order, lq.system_order("musique_full", "x", 20260930))


class TransportTests(unittest.TestCase):
    def test_user_message_is_byte_identical_to_v1_1(self):
        qwen, _ = lq.build_qwen_messages("Q?", EVIDENCE)
        v11, _ = build_messages_v1_1("Q?", EVIDENCE)
        self.assertEqual(qwen[1], v11[1])
        self.assertTrue(qwen[0]["content"].startswith(v11[0]["content"]))
        self.assertIn(lq.SERIALIZED_SCHEMA, qwen[0]["content"])

    def test_fingerprint_is_new(self):
        request = {"dataset": "qasper", "question_id": "q", "system": "sar_rag_v1", "question": "Q?", "evidence": EVIDENCE}
        self.assertNotEqual(lq.qwen_request_fingerprint(request, "abc", 1), request_fingerprint_v1_1(request))
        self.assertNotEqual(lq.qwen_request_fingerprint(request, "abc", 1), lq.qwen_request_fingerprint(request, "abc", 2))


class NoRepairValidationTests(unittest.TestCase):
    def test_infrastructure_test_output_is_schema_invalid_not_repaired(self):
        raw = '{\n  "answerability": "answerable",\n  "claims": {\n    "freezing_temperature": "0 degrees Celsius"\n  },\n  "missing_information": {}\n}'
        result = lq.evaluate_output(raw, 1)
        self.assertEqual(result["validation_status"], "schema_invalid")
        self.assertIsNone(result["validated_response"])
        self.assertEqual(result["parsed_output"]["claims"], {"freezing_temperature": "0 degrees Celsius"})
        self.assertTrue(any("claims is dict" in error for error in result["schema_errors"]))
        self.assertTrue(any("answerability not in enum" in error for error in result["schema_errors"]))
        self.assertTrue(any("missing_information is dict" in error for error in result["schema_errors"]))

    def test_markdown_fence_is_a_parse_failure(self):
        raw = '```json\n{"answerability":"INSUFFICIENT","claims":[],"missing_information":"x"}\n```'
        self.assertEqual(lq.evaluate_output(raw, 0)["validation_status"], "parse_failure")

    def test_duplicate_keys_and_nan_are_rejected(self):
        dup = '{"answerability":"INSUFFICIENT","answerability":"ANSWERABLE","claims":[],"missing_information":""}'
        self.assertEqual(lq.evaluate_output(dup, 1)["validation_status"], "parse_failure")
        nan = '{"answerability":"INSUFFICIENT","claims":[],"missing_information":NaN}'
        self.assertEqual(lq.evaluate_output(nan, 1)["validation_status"], "parse_failure")

    def test_valid_answer(self):
        raw = json.dumps({"answerability": "ANSWERABLE", "claims": [{"text": "0 C.", "evidence_ids": ["E1"]}], "missing_information": ""})
        result = lq.evaluate_output(raw, 2)
        self.assertEqual(result["validation_status"], "valid")
        self.assertEqual(result["validated_response"]["rendered_answer"], "0 C. [E1]")

    def test_semantic_failures(self):
        unknown = json.dumps({"answerability": "ANSWERABLE", "claims": [{"text": "x", "evidence_ids": ["E9"]}], "missing_information": ""})
        self.assertEqual(lq.evaluate_output(unknown, 2)["validation_status"], "semantic_invalid")
        insufficient = json.dumps({"answerability": "INSUFFICIENT", "claims": [{"text": "x", "evidence_ids": ["E1"]}], "missing_information": "gap"})
        self.assertEqual(lq.evaluate_output(insufficient, 2)["validation_status"], "semantic_invalid")
        missing = json.dumps({"answerability": "INSUFFICIENT", "claims": []})
        self.assertEqual(lq.evaluate_output(missing, 2)["validation_status"], "schema_invalid")

    def test_bad_evidence_id_types(self):
        raw = json.dumps({"answerability": "ANSWERABLE", "claims": [{"text": "x", "evidence_ids": [1]}], "missing_information": ""})
        self.assertEqual(lq.evaluate_output(raw, 2)["validation_status"], "schema_invalid")


class CheckpointTests(unittest.TestCase):
    def test_torn_line_is_tolerated_and_meta_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            runner = lq.LocalQwenRunner(directory, {"a": 1})
            lq.append_jsonl(runner.predictions_path, {"logical_key": "k1", "request_fingerprint": "f"})
            with runner.predictions_path.open("a", encoding="utf-8") as stream:
                stream.write('{"logical_key": "k2", "trunc')
            again = lq.LocalQwenRunner(directory, {"a": 1})
            self.assertEqual(set(again.completed), {"k1"})
            self.assertEqual(again.torn_lines, [2])
            lq.append_jsonl(runner.predictions_path, {"logical_key": "k3", "request_fingerprint": "g"})
            self.assertEqual(set(lq.LocalQwenRunner(directory, {"a": 1}).completed), {"k1", "k3"})
            with self.assertRaises(ValueError):
                lq.LocalQwenRunner(directory, {"a": 2})


class ScorePairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib.util
        spec = importlib.util.spec_from_file_location("score_local", ROOT / "tools" / "score_local_qwen_4bit.py")
        cls.score = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.score)

    def record(self, status, response, evidence_text="Water freezes at 0 degrees Celsius at sea level pressure."):
        evidence = [{"evidence_id": "E1", "text": evidence_text}]
        return {"logical_key": "qasper|q|sar_rag_v1", "dataset": "qasper", "question_id": "q", "system": "sar_rag_v1",
                "source_group_id": "g", "validation_status": status, "validated_response": response,
                "retrieved_evidence": [{"text": evidence_text}, {"text": "unrelated"}], "supplied_evidence": evidence,
                "supplied_context_tokens_minilm": 20, "native_context_tokens_minilm": 25, "qwen_supplied_evidence_tokens": 18,
                "final_k": 2, "input_tokens": 100, "output_tokens": 10, "total_tokens": 110,
                "latency_end_to_end_before_save_seconds": 1.0, "latency_generation_seconds": 0.9,
                "throughput_output_tokens_per_second": 11.0, "peak_allocated_vram_gib": 6.0, "peak_reserved_vram_gib": 6.4,
                "output_truncated_at_max_new_tokens": False}

    def test_valid_answerable(self):
        response = {"answerability": "ANSWERABLE", "claims": [{"text": "0 degrees Celsius", "evidence_ids": ["E1"]}]}
        label = {"gold_answerability": "ANSWERABLE", "gold_answers": ["0 degrees Celsius"], "gold_evidence_spans": ["freezes at 0 degrees Celsius"]}
        row = self.score.score_pair(self.record("valid", response), label, None)
        self.assertEqual((row["answer_token_f1"], row["answer_exact_match"], row["answerability_correct"]), (1.0, 1.0, 1.0))
        self.assertEqual((row["supplied_evidence_recall"], row["answerable_complete_evidence"]), (1.0, 1.0))

    def test_invalid_generation_is_failure_not_abstention(self):
        label = {"gold_answerability": "INSUFFICIENT", "gold_answers": [], "gold_evidence_spans": []}
        row = self.score.score_pair(self.record("semantic_invalid", None), label, None)
        self.assertEqual((row["answer_token_f1"], row["answer_exact_match"], row["abstained"], row["correct_abstention"]), (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(row["predicted_answerability"], "GENERATION_FAILURE")

    def test_undefined_gold_answerability_is_nan(self):
        label = {"gold_answerability": None, "gold_answers": [], "gold_evidence_spans": ["freezes at 0 degrees Celsius"]}
        response = {"answerability": "INSUFFICIENT", "claims": []}
        row = self.score.score_pair(self.record("valid", response), label, None)
        self.assertNotEqual(row["answer_token_f1"], row["answer_token_f1"])  # NaN
        self.assertEqual(row["supplied_evidence_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
