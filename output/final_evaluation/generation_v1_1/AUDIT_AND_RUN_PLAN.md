# Smoke-test generation v1.1 audit and run plan

Status: frozen; awaiting explicit paid authorization. No v1.1 API calls have been made.

## Actual v1 outgoing request

The original client sent `POST https://api.groq.com/openai/v1/chat/completions` with JSON fields:

- `model`: `openai/gpt-oss-120b`
- `messages`: one system message and one user message containing the question and frozen evidence
- `temperature`: `0`
- `top_p`: `1`
- `max_completion_tokens`: `800`
- `reasoning_effort`: `medium`
- `response_format.type`: `json_schema`
- `response_format.json_schema.name`: `frozen_final_answer`
- `response_format.json_schema.strict`: `true`
- no `reasoning_format`, `include_reasoning`, tools, streaming, or seed field

The schema required exactly `answerability`, `claims`, and `missing_information` at the root. Claim objects required exactly `text` and `evidence_ids`. Both object levels used `additionalProperties: false`. The semantic validator additionally required non-empty claims and valid citations for answerable outputs and prohibited claims for `INSUFFICIENT`.

## Official-documentation compliance review

Reviewed 2026-09-26:

- https://console.groq.com/docs/structured-outputs
- https://console.groq.com/docs/reasoning
- https://console.groq.com/docs/api-reference
- https://console.groq.com/docs/model/openai/gpt-oss-120b

Groq currently lists GPT-OSS 120B as supporting `strict: true`. The documented mandatory strict-schema rules are that every object property is required and every object is closed with `additionalProperties: false`; v1 satisfied both. `medium` reasoning effort and `max_completion_tokens` are supported. `reasoning_format` is not supported for GPT-OSS and was correctly omitted.

The documentation says strict mode guarantees schema compliance, yet the saved calls returned generated-output validation HTTP 400s. Therefore, the request satisfies the documented mandatory shape, while the observed strict-mode behavior remains an operational provider/model issue.

## Two unresolved pairs

1. `musique_full|musique_full:train:4hop1__38130_8966_31714_41245:answerable|traditional_k10`
2. `musique_full|musique_full:train:4hop1__38130_8966_31714_41245:answerable|sar_rag_v1`

Confirmed:

- Each pair received four attempts across the original run and recovery.
- All eight attempts returned HTTP 400 `json_validate_failed` and produced no validated answer.
- The final SAR-RAG v1 error retained generated JSON containing `answerability` and `claims` but missing the required `missing_information` field.
- The other seven errors, including every traditional-k10 error, did not provide enough generated content to identify the specific schema violation.
- A separate recovery pair explicitly failed with `max completion tokens reached before generating a valid document`.
- Four saved valid v1 responses used at least 700 of 800 completion tokens; two used at least 790.

Not confirmed:

- Neither unresolved pair reported token exhaustion. Token exhaustion is not assigned as their cause.
- The saved evidence does not establish that temperature, reasoning effort, prompt length, the citation regex, or question difficulty caused the failures.
- Seven of the eight unresolved-pair errors cannot be reconstructed because the provider returned an empty `failed_generation`.

Hypotheses retained only as hypotheses:

- Explicitly naming all required fields may reduce model-side omission before provider validation.
- More completion headroom may reduce failures for cases where reasoning and structured content approach the former 800-token ceiling.
- Strict constrained decoding may have provider-side instability on these requests despite documented support.

## Smallest justified v1.1 correction

Unchanged:

- model `openai/gpt-oss-120b`
- temperature `0`
- top-p `1`
- reasoning effort `medium`
- `strict: true`
- JSON schema, including all required fields
- semantic response validator
- all 48 system-question pairs, questions, and frozen evidence

Changed:

1. The prompt explicitly requires all three top-level fields, states that `missing_information` must always be emitted and may be an empty string, requires `claims: []` for `INSUFFICIENT`, and reiterates citation requirements.
2. `max_completion_tokens` increases from 800 to 1600.

No field is inserted or repaired after generation. Semantic validation is unchanged. Invalid responses remain failed attempts. Because prompt and generation settings changed, all 48 pairs must be rerun under v1.1; the prior 46 answers will not be mixed into the v1.1 result.

## Fresh-run rules

- Fresh pairs: 48; no reuse of v1 or recovery answers.
- Maximum attempts per pair: 3 total (initial plus 2 retries).
- Maximum attempts overall: 144, including every HTTP error and local validation failure.
- Round-robin scheduling: every unresolved pair receives attempt N before any pair receives attempt N+1.
- Retryable: HTTP 429, HTTP 5xx, timeout/network failure, generated-output validation/JSON parse failure, and local semantic-validation failure.
- Fatal run abort: HTTP 401/403; non-generation request/configuration rejection; fingerprint, authorization, attempt, or cost guard mismatch.
- Rate limiting: at least 30 seconds between attempt starts; honor longer server retry delay plus 0.25 seconds; every 429 counts.
- Persistent failure: retain all attempts and raw available output, leave pair missing after attempt 3, do not insert fields, repair responses, impute answers, reuse another system's answer, or discard the question.
- Reporting: record first-attempt validity, eventual validity within three attempts, first-valid attempt number, and persistent failures separately.
- Stop early when all 48 pairs validate.

## Cost preflight

- Exact tokenizer input for one 48-pair round: 72,489 tokens.
- Observed provider prompt-usage delta in recovery: 67 tokens per successful request.
- Conservative provider-framing reserve: 128 additional input tokens per attempt, or 78,633 billed-input tokens per round.
- Maximum output per attempt: 1,600 tokens.
- Conservative one-round cost: $0.05787495.
- Conservative three-round/144-attempt ceiling: $0.17362485.
- Assumptions: every attempt consumes its exact preflight input and all 1,600 output tokens; no cached-input discount.
- Before every call, the runner checks remaining attempt allowance and conservative remaining cost allowance.

## Frozen identifiers

- Protocol fingerprint: `c915bed876a3e1394e09d160caa4bffac6c5df60763da0d5186674421c326a79`
- Run-manifest fingerprint: `75a0047c16fa631e2e50ebbe76644867d9d147f94e56d77d3b4fd613b8c11a80`

The held-out evaluation remains prohibited. Smoke outputs are operational evidence only and cannot support research-superiority claims.
