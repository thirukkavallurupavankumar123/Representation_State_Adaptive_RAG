# Exact Paid Execution Plan

**Status:** NOT AUTHORIZED — no paid call has been made during this audit.
**Precondition gate:** every item in §6 must be satisfied before any command in §3 is run.

All token figures below are **measured**, not assumed: they come from
`output/pilot_fair_generator_qasper2/results.json` (2 QASPER questions, one
Traditional + one SAR-RAG + one evaluator call each) and from the retrieval-only
context-token means in the frozen calibration manifest. Their limitations are
stated in §5.

---

## 1. Workload

| Quantity | QASPER test | MuSiQue dev | Total |
|---|---:|---:|---:|
| Questions | 1,451 | 4,834 | 6,285 |
| Traditional answer calls | 1,451 | 4,834 | 6,285 |
| SAR-RAG answer calls | 1,451 | 4,834 | 6,285 |
| Blinded evaluator calls | 1,451 | 4,834 | 6,285 |
| **Total API calls** | **4,353** | **14,502** | **18,855** |

Each *system* makes exactly one online call per question. The evaluator is a
separate research instrument, not part of either system. Three calls per
question is the study cost; one call per question is the method's cost. These
must never be conflated in the paper.

Dropping `--skip-quality-judge` removes 6,285 calls (33%) and loses only the
self-evaluation-biased faithfulness/relevancy pair.

## 2. Measured per-call token usage

From the QASPER pilot (n = 2):

| Call | Prompt tokens | Completion tokens | Cached share |
|---|---:|---:|---:|
| Traditional answer | 3,156 | 659 | 0.0% |
| SAR-RAG answer | 2,999 | 554 | 0.0% |
| Evaluator | 3,911 | 350 | 3.3% |

Prompt caching is effectively absent because every prompt carries different
evidence text. Cost is therefore driven almost entirely by uncached input and
output tokens.

Projected totals for 6,285 questions:

| | Prompt | Completion |
|---|---:|---:|
| Traditional | 19.83 M | 4.14 M |
| SAR-RAG | 18.85 M | 3.48 M |
| Evaluator | 24.58 M | 2.20 M |
| **Total** | **63.26 M** | **9.82 M** |

Without the evaluator: 38.68 M prompt, 7.62 M completion.

## 3. Cost formula

Cost is **undefined** in this report. `GROQ_INPUT_COST_PER_MILLION`,
`GROQ_CACHED_INPUT_COST_PER_MILLION` and `GROQ_OUTPUT_COST_PER_MILLION` are all
absent from the environment, and `experiments.py` correctly refuses to start a
confirmatory run without them. Per the audit rules, token usage is reported and
cost is left undefined until run-date rates are independently verified.

Substitute verified run-date rates (USD per million tokens):

```
cost_usd = 63.26 * R_input + 9.82 * R_output          # with evaluator
cost_usd = 38.68 * R_input + 7.62 * R_output          # without evaluator
```

Worked sensitivity (illustrative arithmetic only — **not** a price quote):

| R_input | R_output | With evaluator | Without evaluator |
|---:|---:|---:|---:|
| 0.10 | 0.40 | $10.25 | $6.92 |
| 0.15 | 0.60 | $15.38 | $10.37 |
| 0.50 | 1.50 | $46.36 | $30.77 |

The rates you supply are frozen into `run_fingerprint.json`, so a resume with
different rates is refused. Verify them against Groq's pricing page on the run
date and record the date in the audit trail.

## 4. Runtime, disk and checkpointing

- **Wall time:** the pilot averaged 17.7 s (Traditional) and 14.4 s (SAR-RAG)
  end-to-end per question, plus the evaluator call. Serially that is roughly
  **56 h** for 6,285 questions, excluding indexing. This is dominated by Groq
  latency, not local compute.
- **Indexing:** one corpus is built per unique document set and only the active
  corpus is retained. QASPER rebuilds per paper (416 unique papers in test);
  MuSiQue rebuilds per question (each carries its own 20 paragraphs).
- **Disk:** one checkpoint JSON per example under `checkpoints/`, each holding
  evidence text, provisional evidence and traces. The pilot's 2-example
  checkpoint was 50 KB, so budget roughly **150–400 MB per dataset**, plus
  `results.json` of comparable size.
- **Interruption:** every example is written atomically (`.json.tmp` then
  `replace`), so a kill loses at most the in-flight example. Resume requires the
  byte-identical command; any change to input, manifest, model, condition,
  static k, cost rates or the proposed config is refused.

Run the two datasets in **separate output directories** so that a failure in one
cannot corrupt the other and so per-dataset results are never pooled by accident.

## 5. Why these numbers are uncertain

1. The pilot is **QASPER-only and n = 2**. MuSiQue supplies ~20 short paragraphs
   per question rather than one long paper, so its retrieval behaviour and
   prompt composition will differ. Treat ±30% as the honest band.
2. Both arms are capped at the same 2,000-token evidence budget, which bounds
   the evidence portion of the prompt. This limits how far the estimate can
   drift upward.
3. Completion tokens depend on how many claims the model emits and are capped at
   1,600 per call.
4. The pilot predates the current `experiments.py`; its records lack
   `complete_evidence_recovery`, `candidates_examined` and the attempt log. Its
   *token* fields are still valid measurements of the same prompt construction.

## 6. Preconditions before any paid call

| # | Gate | Status |
|---|---|---|
| 1 | Direct implementation defects fixed | **DONE** — D01–D07 in `DEFECT_REMEDIATION_LOG.csv` |
| 2 | Tests pass in the project `.venv` | **DONE** — 73 passed under `.venv\Scripts\python.exe` |
| 3 | Version-2 calibration frozen and compared to version 1 | **DONE** — `VERSION1_VS_VERSION2_CALIBRATION.json` |
| 4 | Retrieval-only screening completed | **DONE** — `RETRIEVAL_ONLY_ABLATIONS.csv` |
| 5 | Verified run-date Groq rates supplied and frozen | **BLOCKED** — all three rate variables absent |
| 6 | Explicit authorization from the project owner | **BLOCKED** — not given |
| 7 | Granularity confound (D13) reported before any efficiency claim | **CONTROL AVAILABLE**, must be reported |
| 8 | MuSiQue insufficient-with-gold decision (D14) made | **OPEN** — affects 1,876 examples |

## 7. Commands (do not run until §6 is clear)

```powershell
$env:PYTHONPATH = (Resolve-Path -LiteralPath 'src').Path
$env:GROQ_INPUT_COST_PER_MILLION        = '<verified run-date rate>'
$env:GROQ_CACHED_INPUT_COST_PER_MILLION = '<verified run-date rate>'
$env:GROQ_OUTPUT_COST_PER_MILLION       = '<verified run-date rate>'

# QASPER test - 1,451 questions
.\.venv\Scripts\python.exe -m sar_rag.experiments `
  --input .\datasets\normalized\qasper_test.jsonl `
  --output-dir .\output\independent_verification\confirmatory\qasper `
  --calibration-manifest .\output\calibration\frozen_manifest.json

# MuSiQue development - 4,834 questions
.\.venv\Scripts\python.exe -m sar_rag.experiments `
  --input .\datasets\normalized\musique_full_validation.jsonl `
  --output-dir .\output\independent_verification\confirmatory\musique `
  --calibration-manifest .\output\calibration\frozen_manifest.json
```

Resume after an interruption by appending `--resume` to the **identical**
command. Never pass `--limit`; confirmatory mode rejects it. Never pass
`--static-k`; it is refused against a frozen manifest.

Analysis once both directories exist:

```powershell
.\.venv\Scripts\python.exe -m sar_rag.q2_analysis `
  --results .\output\independent_verification\confirmatory\qasper\results.json `
  --results .\output\independent_verification\confirmatory\musique\results.json `
  --dataset-input .\datasets\normalized\qasper_test.jsonl `
  --dataset-input .\datasets\normalized\musique_full_validation.jsonl `
  --output-dir .\output\independent_verification\q2_analysis `
  --resamples 10000 --failure-sample 100
```

## 8. Staged ablation design

A full end-to-end sweep of all 20 registered conditions would be roughly
19 × the SAR-RAG half of the primary run — on the order of **238,000 additional
calls**. That is not defensible.

**Stage A — retrieval-only screening (0 paid calls, already executed).**
15 of the 20 conditions change only retrieval and are fully screened offline on
training data by `tools/retrieval_only_screening.py`: `full`, the four
`*_only` views, `all_representations_no_rrf`, `rrf_probe_only_no_adaptive`,
`no_novelty`, `no_similarity_gap`, `no_post_retrieval_deduplication`,
`select_one_representation`, `probe_depth_5`, and `rrf_k0_{1,10,60}`. The four
`baseline_fixed_k_*` conditions are screened as the baseline-k sensitivity
sweep. Only `legacy_multi_call` cannot be screened offline, because it changes
the online LLM policy by design.

**Stage B — preregistered promotion rule.** Promote at most four conditions,
chosen by *scientific necessity* and fixed before any held-out number is seen:

1. `legacy_multi_call` — the only condition that cannot be screened offline, and
   the sole evidence for the one-call cost claim. **Mandatory.**
2. `select_one_representation` — isolates the paper's central "one *or two*
   representations" claim. **Mandatory**, and especially so given the frozen
   threshold selects two views in a small minority of queries (D16).
3. `all_representations_no_rrf` — isolates whether RRF *selection* matters at
   all versus simply using every view. **Mandatory**, as it is the direct null
   for the routing contribution.
4. One granularity control — the 95-word matched sliding-window baseline — to
   separate chunk size from adaptive stopping in the end-to-end efficiency
   claim (D13). **Mandatory** before any token-reduction claim.

This rule is fixed by scientific role, not by screening or held-out
performance. Conditions whose retrieval-only screening shows no material
difference still may not be promoted *in place of* the four above; screening
informs interpretation, never promotion.

**Stage B cost.** Four conditions × 6,285 questions × 1 SAR-RAG call (baseline
and evaluator results are reused from the primary run where the condition does
not change them) ≈ **25,140 additional calls**, roughly 40% of the primary run's
call volume. Run each condition into its own output directory; the condition is
already part of the run fingerprint.

**Stage C.** Report every screened condition from Stage A as retrieval-only
evidence, clearly labelled as such, and never present a retrieval-only result as
an end-to-end ablation.
