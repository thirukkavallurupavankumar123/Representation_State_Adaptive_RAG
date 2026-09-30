# Paid smoke recovery plan

Status: frozen locally; awaiting separate user authorization. No recovery API calls have been made.

## Original evidence integrity

The original protocol, manifests, requests, checkpoint metadata, 48-attempt ledger, 40 validated answers, and 8 failure records were hashed before recovery work. Their hashes are recorded in `ORIGINAL_INTEGRITY.json` and embedded in `RECOVERY_MANIFEST.json`. The recovery runner refuses execution if any original hash changes and refuses to use the original checkpoint as its output directory.

## Diagnosis

- HTTP 429 (one attempt): the response reported an 8,000 TPM limit, 7,513 tokens already used, and 1,689 requested. That pair later produced a validated answer and is not currently missing. Recovery uses 30-second minimum start-to-start pacing, honors a longer server retry delay when present, and counts every 429 against the ceiling.
- HTTP 400 `json_validate_failed` (five pairs): every response had an empty `failed_generation`, so the malformed/truncated generation cannot be reconstructed. The request did use `strict: true` with a model that Groq documents as supporting strict structured output. Among the 40 valid responses, four used at least 700 of the 800 allowed completion tokens and two used at least 790; one sibling response for a failed question used 796 and another used 708. Exhaustion of the shared reasoning/completion budget is therefore a plausible contributor, but the provider's empty failure payload makes this an inference rather than proof.
- Local semantic validation (two pairs): the static JSON schema allows an empty/unknown citation list and does not condition `claims` on `answerability`. The stricter local validator rejected one empty/invalid citation and one `INSUFFICIENT` response containing factual claims. The original runner did not retain the raw rejected payloads, so their exact content is unrecoverable. The recovery runner retains a hash and raw payload for future local-validation failures and retries rather than accepting them.
- Scheduling loss (one pair): the exhausted global ceiling contained one repeat attempt after the 429, so only 47 unique pairs could be attempted. Missing-pair selection is now derived from validated answers rather than from the attempt ledger.

## Exact missing pairs

| # | Dataset | System | Question ID | Original state | Input tokens |
|---:|---|---|---|---|---:|
| 1 | musique_full | traditional_k10 | `musique_full:train:4hop1__38130_8966_31714_41245:answerable` | HTTP 400 JSON validation | 1,177 |
| 2 | musique_full | sar_rag_v1 | `musique_full:train:4hop1__38130_8966_31714_41245:answerable` | HTTP 400 JSON validation | 1,708 |
| 3 | musique_full | traditional_k10 | `musique_full:train:2hop__838053_467995:insufficient` | Local invalid/missing citation | 852 |
| 4 | musique_full | sar_rag_v1 | `musique_full:train:4hop1__493923_300471_136629_85446:insufficient` | HTTP 400 JSON validation | 1,882 |
| 5 | musique_full | adaptive_single_view_v2 | `musique_full:train:4hop1__493923_300471_136629_85446:insufficient` | HTTP 400 JSON validation | 1,882 |
| 6 | qasper | traditional_k10 | `qasper:train:b24767fe7e6620369063e646fd3048dc645a8348` | Local `INSUFFICIENT` with claims | 1,178 |
| 7 | qasper | adaptive_single_view_v2 | `qasper:train:3be8859103016ce2afe4c0a8552b9d980f7958bf` | HTTP 400 JSON validation | 1,776 |
| 8 | qasper | adaptive_single_view_v2 | `qasper:train:2ec640e6b4f1ebc158d13ee6589778b4c08a04c8` | Never attempted | 1,824 |

## Recommended protocol-preserving recovery

Use the original frozen v1 prompt, static schema, model `openai/gpt-oss-120b`, temperature 0, top-p 1, medium reasoning, and 800-token maximum. Only the eight missing pairs are eligible.

- Attempt order: round-robin, so all eight receive one attempt before any pair receives a second or third.
- Maximum per pair: 3 attempts including the initial attempt (2 retries).
- Hard global maximum: 24 API attempts including all HTTP errors and local-validation failures.
- Rate control: at least 30 seconds between attempt starts and any longer server-provided retry delay is honored.
- Exact tokenizer preflight: 12,279 input tokens for one pass across the eight pairs.
- Worst case at one attempt per pair: $0.00568185.
- Authorized ceiling proposed for three complete rounds: $0.01704555.
- Output: a new `recovery_v1/checkpoint`; the original checkpoint is never appended to or rewritten.
- Stop condition: stop early when all eight responses validate; otherwise stop at 24 attempts and report remaining gaps.

This recovery does not change the answer-generation configuration, so a consistent rerun of the existing 40 valid pairs is not required.

## Versioned contingency

If the protocol-preserving recovery cannot validate all eight pairs, do not mix altered outputs into the v1 smoke. Freeze a v1.1 protocol with a larger completion budget and an explicitly strengthened prompt/request-specific citation schema. Because prompt, schema, or decoding changes can affect the answer distribution, v1.1 would require a fresh consistent rerun of all 48 pairs into a new checkpoint. Its exact attempt and cost ceiling must be preflighted and separately authorized after the v1.1 files are frozen.

## Remaining operational issues

- Groq's strict structured-output documentation says strict mode should be schema-conformant, but five observed calls returned schema-validation HTTP 400s. Retrying is necessary until the provider behavior is resolved.
- The original two rejected raw payloads cannot be recovered from the saved files.
- Account-wide concurrent Groq usage could still cause 429s; such attempts remain chargeable against the proposed ceiling.
- This smoke remains operational evidence only. No held-out execution or research-performance inference is permitted.
