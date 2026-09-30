"""End-to-end contract tests for the paired evaluation harness.

No result artifact in the repository was produced by the current ``experiments``
module: there is no ``run_fingerprint.json`` and no ``checkpoints/`` directory
anywhere under ``output/``.  The atomic-checkpoint, fingerprint-gated resume,
API-attempt-logging and systems-measurement corrections were therefore present
in source but never exercised.  These tests drive the whole harness against a
stubbed Groq transport so those contracts are verified without paid calls.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from sar_rag import experiments


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FROZEN_MANIFEST = REPOSITORY_ROOT / "output" / "calibration" / "frozen_manifest.json"

RATES = ["--input-cost-per-million", "0.15",
         "--cached-input-cost-per-million", "0.075",
         "--output-cost-per-million", "0.60"]


class StubGroqClient:
    """Mimics GroqClient's observable surface without any network access."""

    instances: list["StubGroqClient"] = []

    def __init__(self, model: str = "openai/gpt-oss-120b", api_key: str | None = None,
                 timeout_seconds: int = 90, max_retries: int = 8):
        self.model = model
        self.usage_log: list[dict] = []
        self.attempt_log: list[dict] = []
        StubGroqClient.instances.append(self)

    def complete(self, messages, schema_name=None, schema=None, max_tokens=2048) -> str:
        prompt_tokens = sum(len(str(m.get("content", "")).split()) for m in messages)
        self.usage_log.append({
            "model": self.model, "latency_seconds": 0.01,
            "prompt_tokens": prompt_tokens, "completion_tokens": 20,
            "total_tokens": prompt_tokens + 20, "cached_tokens": 5,
            "rate_limit_remaining_requests": None, "rate_limit_remaining_tokens": None,
        })
        self.attempt_log.append({"status": "success", "attempt": 1, "latency_seconds": 0.01})
        if schema_name == "paired_rag_quality":
            return json.dumps({"A": {"faithfulness": 0.9, "answer_relevancy": 0.8},
                               "B": {"faithfulness": 0.7, "answer_relevancy": 0.6}})
        return json.dumps({
            "answerability": "ANSWERABLE",
            "claims": [{"text": "Officers retain documentary evidence.",
                        "evidence_ids": ["E1"]}],
            "missing_information": "",
        })


def example_row(index: int) -> dict:
    body = "\n\n".join(
        part
        for section in range(4)
        for part in ([f"# Section {section} retention rules"] +
                     [f"Officers in category {section} retain documentary evidence of "
                      f"decision {paragraph} for seven years and record the outcome "
                      f"in the register of determinations."
                      for paragraph in range(3)])
    )
    return {
        "example_id": f"stub:test:{index}",
        "dataset": "stub", "split": "test",
        "query": "What must officers retain?",
        "documents": [{
            "document_id": f"doc{index}", "version_id": "v1",
            "title": "Retention policy", "text": body,
            "effective_date": None, "authority_level": "policy",
            "supersedes_version_id": None,
        }],
        "gold_answers": ["documentary evidence"],
        "gold_evidence_spans": [
            "Officers in category 0 retain documentary evidence of decision 0 for "
            "seven years and record the outcome in the register of determinations."
        ],
        "gold_answerability": "ANSWERABLE",
        "metadata": {"source_id": "2hop_stub"},
    }


class ExperimentHarnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original_client = experiments.GroqClient
        experiments.GroqClient = StubGroqClient
        StubGroqClient.instances = []
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.input_path = self.root / "input.jsonl"
        self.input_path.write_text(
            "\n".join(json.dumps(example_row(index)) for index in range(3)) + "\n",
            encoding="utf-8",
        )
        import os
        self._had_key = os.environ.get("GROQ_API_KEY")
        os.environ["GROQ_API_KEY"] = "stub-key-not-a-real-credential"

    def tearDown(self) -> None:
        import os
        experiments.GroqClient = self._original_client
        if self._had_key is None:
            os.environ.pop("GROQ_API_KEY", None)
        else:
            os.environ["GROQ_API_KEY"] = self._had_key
        self._temporary.cleanup()

    def _run(self, output_dir: Path, extra: list[str] | None = None) -> int:
        return experiments.main([
            "--input", str(self.input_path),
            "--output-dir", str(output_dir),
            "--calibration-manifest", str(FROZEN_MANIFEST),
            "--bootstrap-resamples", "200",
            *RATES, *(extra or []),
        ])

    def test_confirmatory_run_writes_fingerprint_checkpoints_and_systems_fields(self) -> None:
        output = self.root / "run1"
        self.assertEqual(self._run(output), 0)

        fingerprint = json.loads((output / "run_fingerprint.json").read_text(encoding="utf-8"))
        self.assertEqual(fingerprint["condition"], "full")
        self.assertEqual(fingerprint["static_k"], 20)
        self.assertFalse(fingerprint["pilot"])
        self.assertIsNone(fingerprint["limit"])
        self.assertEqual(fingerprint["cost_rates_per_million_usd"], [0.15, 0.075, 0.60])
        # The frozen controls must appear in the fingerprint, not the defaults.
        self.assertEqual(fingerprint["proposed_config"]["similarity_threshold"], 0.5289)
        self.assertEqual(fingerprint["proposed_config"]["empty_batch_patience"], 3)

        checkpoints = sorted((output / "checkpoints").glob("*.json"))
        self.assertEqual(len(checkpoints), 3)
        from hashlib import sha256
        for index in range(3):
            expected = sha256(f"stub:test:{index}".encode()).hexdigest() + ".json"
            self.assertTrue((output / "checkpoints" / expected).is_file())
        self.assertFalse(list((output / "checkpoints").glob("*.tmp")))

        payload = json.loads((output / "results.json").read_text(encoding="utf-8"))
        summary = payload["summary"]
        self.assertEqual(summary["status"], "confirmatory")
        self.assertEqual(summary["examples"], 3)

        systems = summary["offline_indexing"]
        for field in ("traditional_index_vector_bytes", "proposed_index_vector_bytes",
                      "traditional_indexed_chunks", "proposed_indexed_chunks",
                      "traditional_embedding_calls", "proposed_embedding_calls",
                      "traditional_embedded_texts", "proposed_embedded_texts",
                      "process_peak_working_set_bytes", "unique_corpora_built_this_invocation"):
            self.assertIn(field, systems)
            self.assertIsNotNone(systems[field], field)
            self.assertGreater(systems[field], 0, field)
        self.assertGreater(systems["proposed_indexed_chunks"], systems["traditional_indexed_chunks"])

        self.assertEqual(summary["api_attempts"]["traditional"], {"success": 3})
        self.assertEqual(summary["api_attempts"]["proposed"], {"success": 3})
        self.assertEqual(summary["api_attempts"]["evaluator"], {"success": 3})
        self.assertIn("exact_mcnemar_tests", summary)
        self.assertIn("holm_adjusted_primary_p_values", summary)
        self.assertIn("predeclared_practical_margins", summary)
        self.assertEqual(summary["cost_rates_per_million_usd"],
                         {"uncached_input": 0.15, "cached_input": 0.075, "output": 0.60})

    def test_exactly_one_answer_call_per_system_and_one_evaluator_call(self) -> None:
        output = self.root / "run_calls"
        self._run(output)
        payload = json.loads((output / "results.json").read_text(encoding="utf-8"))
        for record in payload["records"]:
            self.assertEqual(record["traditional"]["api_calls"], 1.0)
            self.assertEqual(record["proposed"]["api_calls"], 1.0)
            self.assertEqual(record["evaluator_usage"]["api_calls"], 1.0)
            self.assertEqual(record["traditional"]["api_attempts"], 1.0)
            self.assertEqual(record["proposed"]["failed_api_attempts"], 0.0)
            self.assertGreater(record["traditional"]["estimated_cost_usd"], 0.0)
        # Three questions x (SAR + Traditional + evaluator) = nine calls total.
        total = sum(len(client.usage_log) for client in StubGroqClient.instances)
        self.assertEqual(total, 9)

    def test_per_record_traces_and_provisional_evidence_are_persisted(self) -> None:
        output = self.root / "run_trace"
        self._run(output)
        payload = json.loads((output / "results.json").read_text(encoding="utf-8"))
        for record in payload["records"]:
            self.assertIn("proposed", record["trace_summary"])
            self.assertIn("stop_reason", record["trace_summary"]["proposed"])
            self.assertIn("rejection_counts", record["trace_summary"]["proposed"])
            self.assertIn("representation_profiles", record["trace_summary"]["proposed"])
            self.assertIn("proposed", record["provisional_evidence"])
            self.assertIn("proposed", record["evidence"])
            self.assertTrue(record["selected_representations"])
            self.assertLessEqual(len(record["selected_representations"]), 2)
            self.assertIn("representation_gold_diagnostic", record)
            self.assertIn("descriptors", record)
            self.assertEqual(record["descriptors"]["hop_count"], 2)
            # Traditional now reports the candidates it actually examined, which is
            # min(static_k, index size) and never the smaller post-budget final_k.
            examined = record["traditional"]["candidates_examined"]
            self.assertGreaterEqual(examined, record["traditional"]["final_k"])
            self.assertLessEqual(examined, 20.0)
            self.assertGreater(examined, 0.0)
            self.assertGreaterEqual(record["proposed"]["retrieval_rounds"], 0.0)

    def test_confirmatory_mode_rejects_limit_and_baseline_k_override(self) -> None:
        with self.assertRaises(SystemExit):
            self._run(self.root / "run_limit", ["--limit", "1"])
        with self.assertRaises(SystemExit):
            self._run(self.root / "run_k", ["--static-k", "5"])

    def test_confirmatory_mode_requires_manifest_and_cost_rates(self) -> None:
        with self.assertRaises(SystemExit):
            experiments.main(["--input", str(self.input_path),
                              "--output-dir", str(self.root / "run_nomanifest"), *RATES])
        with self.assertRaises(SystemExit):
            experiments.main(["--input", str(self.input_path),
                              "--output-dir", str(self.root / "run_norates"),
                              "--calibration-manifest", str(FROZEN_MANIFEST)])

    def test_rerun_without_resume_is_refused(self) -> None:
        output = self.root / "run_rerun"
        self._run(output)
        with self.assertRaises(SystemExit):
            self._run(output)

    def test_resume_requires_an_exact_fingerprint_match(self) -> None:
        output = self.root / "run_resume"
        self._run(output)
        # A different condition changes the fingerprint and must be refused.
        with self.assertRaises(SystemExit):
            self._run(output, ["--resume", "--condition", "no_novelty"])
        # Different cost rates also change the fingerprint.
        with self.assertRaises(SystemExit):
            experiments.main([
                "--input", str(self.input_path), "--output-dir", str(output),
                "--calibration-manifest", str(FROZEN_MANIFEST), "--resume",
                "--bootstrap-resamples", "200",
                "--input-cost-per-million", "9.99",
                "--cached-input-cost-per-million", "0.075",
                "--output-cost-per-million", "0.60",
            ])

    def test_resume_reuses_checkpoints_and_makes_no_further_calls(self) -> None:
        output = self.root / "run_resume_ok"
        self._run(output)
        calls_after_first = sum(len(c.usage_log) for c in StubGroqClient.instances)
        self.assertEqual(calls_after_first, 9)
        self.assertEqual(self._run(output, ["--resume"]), 0)
        new_clients = StubGroqClient.instances[-3:]
        self.assertEqual(sum(len(c.usage_log) for c in new_clients), 0,
                         "resume re-issued API calls for completed examples")
        payload = json.loads((output / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["summary"]["examples"], 3)
        self.assertEqual(len({r["example_id"] for r in payload["records"]}), 3)

    def test_resume_without_a_prior_run_is_refused(self) -> None:
        with self.assertRaises(SystemExit):
            self._run(self.root / "run_missing", ["--resume"])

    def test_registered_condition_is_recorded_in_the_fingerprint(self) -> None:
        output = self.root / "run_condition"
        self._run(output, ["--condition", "no_similarity_gap"])
        fingerprint = json.loads((output / "run_fingerprint.json").read_text(encoding="utf-8"))
        self.assertEqual(fingerprint["condition"], "no_similarity_gap")
        self.assertFalse(fingerprint["proposed_config"]["similarity_gap_enabled"])
        payload = json.loads((output / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["summary"]["registered_condition"], "no_similarity_gap")


if __name__ == "__main__":
    unittest.main()
