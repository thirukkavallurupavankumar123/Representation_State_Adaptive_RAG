# Final evaluation protocol amendment v1.1

Status: **frozen; no held-out generation or scoring authorized**.

This amendment preserves the original final-evaluation protocol and adopts the smoke-validated v1.1 generation configuration for Traditional RAG, frozen SAR-RAG v1, and frozen adaptive single-view v2. The answer model is `openai/gpt-oss-120b`; temperature is 0, top-p is 1, reasoning effort is medium, strict JSON-schema mode is enabled, and the completion limit is 1,600 tokens. The prompt, schema, semantic validator, and decoding settings are identical across systems.

The frozen retrieval manifests are unchanged. SAR-RAG v1 and adaptive v2 retain their native dynamic final `k`. A deterministic post-retrieval packer enforces the common 1,600-token evidence cap without truncating chunks. Native retrieved evidence and actually supplied evidence must be logged separately, so packing cannot silently redefine either adaptive retrieval policy.

Held-out request fingerprints bind the dataset, question ID, system, evidence, generation configuration, and exact question text. This strengthens checkpoint identity without changing the smoke-validated generation configuration.

## Eligible evaluation population

The exposure audit starts from QASPER official test and MuSiQue-Full official development. It excludes 43 QASPER test questions whose normalized wording exactly matches exposed training questions and both variants of one MuSiQue development source group used in an earlier pilot. The frozen population is therefore:

- QASPER test: 1,408 questions across 415 paper clusters.
- MuSiQue-Full development: 4,832 questions across 2,416 source groups.
- Total: 6,240 questions and 18,720 system-question pairs.

All final IDs had previously been enumerated in an ID-only boundary registry. That enumeration did not expose answer labels, evidence labels, or outcomes, but the IDs are not literally unseen. QASPER test remains eligible for dataset-specific test claims after the exclusions. MuSiQue remains eligible only as development-split evidence, not as a hidden-test result. Public-model pretraining exposure cannot be ruled out.

## Fair comparison and outcomes

Traditional RAG at predefined `k=10` is the only generation baseline; no post-hoc baseline or extra generation arm is allowed. Report answer token F1, Exact Match, native and supplied unique gold-evidence recall, answerable-only complete-evidence recovery, insufficient-question residual-support recovery, answerability and abstention metrics, native and supplied context tokens, provider billed tokens, latency, estimated cost, and dynamic final `k`. Report paired quality differences jointly with actual token differences and Pareto relationships.

Faithfulness is separate. Citation validity, citation coverage, and lexical support are diagnostics rather than semantic faithfulness proof. No judge-model call is authorized.

Use 10,000-replicate paired cluster bootstrap confidence intervals and paired cluster randomization tests with seed 20260926. Cluster QASPER by paper and MuSiQue by source group. Apply Holm correction across the three system comparisons within each dataset-by-primary-metric family. Report effect sizes, confidence intervals, raw and adjusted p-values, valid question counts, cluster counts, and answerability subgroup counts. Do not average unrelated dataset outcomes into one quality score.

## Failures and authorization

Each pair permits at most three attempts including the first. Retries are round-robin and restricted to declared transient, provider-output, parsing, or semantic-validation failures. Every 429 and failed attempt counts toward both attempt and operational-cost accounting. Persistent failures remain in the denominator: answer F1 and EM are zero, answerability is incorrect rather than abstention, and sealed retrieval metrics remain scorable. Silent exclusions, answer repair, configuration changes, selective reruns, and cross-system answer reuse are prohibited.

The exact machine-readable specification is `FINAL_EVALUATION_PROTOCOL_AMENDMENT_V1_1.json`. Its protocol fingerprint is `e4836e675a057cee20856d58f07db5f927340cfa5926fb2905e93880504dc893`.
