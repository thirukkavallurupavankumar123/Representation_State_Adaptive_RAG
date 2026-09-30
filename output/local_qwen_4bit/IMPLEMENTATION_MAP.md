# Repository implementation map (Phase 0 forensics)

Prepared 2026-09-30 on branch `local-qwen-reduced-eval` (created from `main@4fddbcd`, clean tree).
Paths are relative to the repository root. Nothing listed here was modified by the local-Qwen work.

## Frozen retrieval (reused via sealed outputs, not re-implemented)

| Component | Location | Notes |
|---|---|---|
| Traditional RAG k=10 | `tools/prepare_final_heldout_requests_v1_1.py:132` (`indexes["sentence_window"].search(query, 1, 10, "static")`) over `src/sar_rag/index.py:11` `ExactVectorIndex` | static top-10 sentence-window exact cosine search |
| SAR-RAG v1 | `src/sar_rag/adaptive_pipeline.py:26` `SarRagSystem.ask` with `SarRagConfig(**output/calibration/frozen_manifest.json["frozen_configuration"]["proposed"])` | dynamic final k |
| Adaptive single-view v2 | `src/sar_rag/adaptive_single_view_v2.py:100` `AdaptiveSingleViewV2System.retrieve` with `FROZEN_V2_MANIFEST.json["selected_configuration"]` | paragraph view only; dynamic final k |
| Embeddings | `src/sar_rag/encoding.py:53` `SentenceTransformerEncoder` (all-MiniLM-L6-v2 @ `1110a243…`, mean pooling + L2) | |
| Corpus/views | `src/sar_rag/corpus.py:28`, `src/sar_rag/representations.py:221` | four representations |
| Evidence serialization | `src/sar_rag/generation_v1_1.py:27` `build_messages_v1_1` (`[E#] text` blocks) | |
| 1,600-token supplied-context policy | `tools/prepare_final_heldout_requests_v1_1.py:82` `pack()` | **counts MiniLM tokenizer tokens** (`encoder.token_count`); complete chunks in native order; never truncates |
| Dynamic final k | `native_final_k` in `RETRIEVAL_AND_PACKING_AUDIT.jsonl`; checked by `tools/seal_heldout_requests_v1_1.py:136` | |
| Sealed retrieval outputs | `output/final_evaluation/heldout_v1_1/REQUESTS.jsonl`, `RETRIEVAL_AND_PACKING_AUDIT.jsonl` (Git LFS) | SHA-256 bound in `REQUEST_MANIFEST_SEALED.json` |

## Generation (Groq v1.1, preserved)

| Component | Location |
|---|---|
| Smoke v1.1 prompt / schema | `src/sar_rag/generation_v1_1.py` (prompt `final-evaluation-answer-v1.1`); `ANSWER_SCHEMA` in `src/sar_rag/final_evaluation.py:18` |
| Output validation | `src/sar_rag/final_evaluation.py:124` `validate_answer` |
| Groq request generation | `build_chat_payload_v1_1` (`generation_v1_1.py:56`); `tools/run_smoke_generation_v1_1.py`; `tools/run_final_heldout_v1_1.py` |
| Groq fingerprints | `request_fingerprint_v1_1` (`generation_v1_1.py:92`), `heldout_request_fingerprint_v1_1` (`src/sar_rag/heldout_v1_1.py:22`) |
| Checkpoint/resume | `src/sar_rag/heldout_v1_1.py:30` `HeldoutV11Runner`, `src/sar_rag/smoke_v1_1.py` |
| Smoke v1.1 result | `output/final_evaluation/generation_v1_1/` (48/48 first-attempt valid, `PASSED_OPERATIONAL_SMOKE_V1_1`) |

## Population, exposure, grouping

| Component | Location |
|---|---|
| Exposure audit / exclusions | `tools/prepare_final_heldout_v1_1.py:111` `prior_exposure_exclusions`, `:252` `exposure_report`; result `heldout_v1_1/DATASET_ELIGIBILITY_AND_EXPOSURE.json` |
| Eligible pool (gold-free) | `heldout_v1_1/GENERATION_QUESTIONS.jsonl`: 1,408 QASPER test + 4,832 MuSiQue-Full dev |
| QASPER / MuSiQue loaders | `src/sar_rag/dataset_adapters.py:33` `iter_qasper`, `:102` `iter_musique_full` |
| Groups | `qasper-paper:<paper_id>`, `musique-group:<source_id>` (answerable + insufficient variants; exactly 2 per group) (`prepare_final_heldout_v1_1.py:189-203`) |
| Design-exposed registries | `output/adaptive_single_view_v2/DESIGN_EXPOSED_IDS.json`, `DESIGN_EXPOSED_TRAIN_IDS.json`, `HELDOUT_ID_REGISTRY.json` |
| Held-out v1.1 protocol | `heldout_v1_1/FINAL_EVALUATION_PROTOCOL_AMENDMENT_V1_1.json` (fingerprint `e4836e67…`); generation never executed (fake-answerer dry run only) |

## Scoring and statistics (held-out v1.1, unexecuted)

| Component | Location |
|---|---|
| Answer F1 | `src/sar_rag/final_evaluation.py:286` `token_f1` |
| Exact Match, span coverage | `tools/score_final_heldout_v1_1.py:40` `exact_match`, `:26` `covered` |
| Evidence metrics | same file `:165-181` (native/supplied recall, answerable complete evidence, insufficient residual support) |
| Abstention metrics | same file `:174-177` |
| Cluster bootstrap + sign randomization | same file `:74` `paired_cluster_inference` (10,000 replicates, seed 20260926) |
| Multiple comparisons | same file `:112` `holm` (per dataset x primary metric, 3 comparisons) |
| Older pilot statistics | `src/sar_rag/experiments.py:283` `holm_adjust`, `src/sar_rag/q2_analysis.py` (not the held-out design) |

## Findings that shaped the local-Qwen design

1. The supplied-context cap is measured in **MiniLM** tokens, not in the Groq generator tokenizer, so the cap does not depend on the answer model. No repacking is needed; Qwen token counts are logged in addition.
2. Four held-out artifacts are Git-LFS pointers in the checkout (`git-lfs` is not installed). The objects were fetched from the remote into an ignored cache and verified against the pointer OIDs, which equal the sealed manifest SHA-256 values. The pointer files were not touched.
3. Recorded SHA-256 values of text artifacts were computed over CRLF files on Windows, while git stores LF. All checked artifacts reproduce their recorded hashes as CRLF-equivalent bytes (`manifests/PRIOR_ARTIFACT_INTEGRITY.json`).
4. Raw datasets and the MiniLM snapshot are absent (gitignored). Gold is only reachable after sealing, via the official archives, and must reproduce the normalized-file hashes recorded in the pool.
5. `score_final_heldout_v1_1.py` would raise `KeyError` on QASPER rows without `gold_answerability`. The local protocol predeclares NaN handling for those rows.
6. Groq strict mode sent the schema through `response_format`. Locally, the identical schema is rendered as text in the system message. There is no constrained decoding and no claim of strict enforcement.

## New files (local-Qwen study only)

- `src/sar_rag/local_qwen_4bit.py`: backend (load/placement checks, chat-template transport, seeds, strict no-repair parser/validator, fingerprints, checkpointed one-attempt runner).
- `tools/local_qwen_4bit.py`: `fetch-inputs`, `verify-integrity`, `smoke`, `freeze`, `generate`, `seal`.
- `tools/score_local_qwen_4bit.py`: post-seal `fetch-gold`, `score`.
- `tests/test_local_qwen_4bit.py`: seed, transport, no-repair validation, checkpoint tests.
- `output/local_qwen_4bit/**`: all study artifacts.
