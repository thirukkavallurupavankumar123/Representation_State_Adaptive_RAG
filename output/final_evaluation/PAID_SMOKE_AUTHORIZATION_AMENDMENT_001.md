# Paid smoke authorization amendment 001

The user authorized the paid smoke test with an absolute ceiling of **48 API attempts including retries** and explicitly prohibited held-out evaluation.

The smoke design already contains 48 system-question pairs. Therefore this run uses **zero automatic retries**. Each pair can consume at most one attempt, and a persistent attempt ledger prevents cumulative execution or resume from exceeding 48 attempts.

All scientific and generation settings remain unchanged: `openai/gpt-oss-120b`, temperature 0, top-p 1, medium reasoning, strict JSON schema, 800 maximum output tokens, and the frozen common evidence cap.

The official model tokenizer and chat template counted 68,649 input tokens across all 48 prepared requests, with a maximum of 1,988 for one request. At the currently verified official rates, the maximum cost with 800 output tokens on every request is **$0.03333735**, below the authorized **$0.04320** ceiling.

This authorization does not permit the final held-out experiment or optional LLM-judge stage.
