# SAR-RAG Independent Verification — Audit Report

**Auditor role:** independent senior research engineer / RAG specialist / statistician / reproducibility auditor
**Repository:** `research_projectt` @ `5cb4cd7dc85f9c4bd87f22cfa840494d98b6a9d5`
**Interpreter:** `.venv\Scripts\python.exe` (Python 3.12.7)
**Paid Groq calls made during this audit:** **zero**

## Evidence labelling

Every statement in this report carries one of these labels. The same labels are
used in the PDF.

| Label | Meaning |
|---|---|
| **[REPRODUCED FACT]** | I re-derived it from source code or recorded artifacts in this audit |
| **[CALIBRATION DIAGNOSTIC]** | Training-only calibration output; not an evaluation result |
| **[ENGINEERING PILOT]** | From the 2-question QASPER pilot; not inferential |
| **[CONFIRMATORY RESULT]** | From a completed held-out run — **none exist** |
| **[PENDING]** | Not executed |
| **[REVIEWER INFERENCE]** | My judgement, flagged as such |
| **[LIMITATION]** | A constraint on what can be concluded |

---

## 1. Headline findings

**1. No result artifact in this repository was produced by the current code.**
**[REPRODUCED FACT]** There is no `run_fingerprint.json` and no `checkpoints/`
directory anywhere under `output/`. The pilot at
`output/pilot_fair_generator_qasper2/` carries a singular `checkpoint.json` and
its records lack `complete_evidence_recovery` (a declared primary outcome),
`gold_evidence_spans`, `candidates_examined`, `retrieval_rounds`, `stop_reason`,
`trace_summary`, `provisional_evidence`, `api_attempts`, and the
McNemar/Holm/practical-margin blocks. Corrections **#12, #13, #15, #16** were
therefore present in source but **never executed**. They are now exercised by
`tests/test_experiment_contracts.py`.

**2. The frozen calibration is fully reproducible.** **[REPRODUCED FACT]** An
independent re-run reproduced `output/calibration/frozen_manifest.json`
**bit-for-bit**: all 32 proposed trials, all 6 baseline trials, sample IDs,
partitions, population strata, source hashes and every chosen metric. Re-running
after defect remediation produced identical results again. No threshold changed.
See `VERSION1_VS_VERSION2_CALIBRATION.json`.

**3. Three genuine implementation defects were found and fixed**, one of which
falsifies a claimed correction:

- Peak working set was **always null on 64-bit Windows** (truncated `HANDLE`),
  so correction #16's "OS-reported process peak working set is now recorded" was
  false as shipped.
- Retrieval recall still returned **1.0 when no gold evidence exists** in a
  second live code path, the exact bug correction #5 claims was fixed.
- The baseline reported **`final_k` as candidates examined**, understating its
  own retrieval work.

Full list with severity and verification: `DEFECT_REMEDIATION_LOG.csv` (20
entries across three classes).

**4. The headline efficiency claim is confounded by chunk size.**
**[REVIEWER INFERENCE]** The four adaptive views are bounded to ~95 words; the
baseline sliding window is 256. A context-token reduction therefore mixes
granularity with adaptive stopping. A matched-granularity control was added.
No token-reduction figure may be attributed to adaptive retrieval until it is
reported.

**5. Representation selection may be doing very little.**
**[CALIBRATION DIAGNOSTIC]** The frozen manifest records
`selected_representations = 1.125`, implying roughly **12.5% two-view
selection**. If this holds out, the "one *or two* complementary representations"
mechanism is exercised on a small minority of queries.

**6. Leakage is materially better than the shipped audit implies, once measured
correctly.** **[REPRODUCED FACT]** The shipped audit compares train vs test
wholesale (72 overlapping questions). But only the **60 calibration examples**
were ever fitted. That check did not exist; I ran it. The calibration sample
shares **0 example IDs, 0 source IDs and 0 document IDs** with either evaluation
split, and **1** generic question template touching **2 of 1451** QASPER test
examples (0.14%), **0** with MuSiQue.

---

## 2. Phase 1 — Repository and dependencies

| Check | Result |
|---|---|
| `git status` | clean at `5cb4cd7`; no user files deleted or reset |
| Package layout | `src/sar_rag` with `pyproject.toml` `packages.find where=["src"]` |
| `.env` excluded | **[REPRODUCED FACT]** untracked; `.gitignore` lists `.env`, `.env.*`, negated `!.env.example` |
| API key in tracked files | **[REPRODUCED FACT]** none; regex scan over all 86 tracked files found no `gsk_…` |
| `pip check` | **No broken requirements found** |
| Env vars | `GROQ_API_KEY` **present**, `GROQ_MODEL` **present** (`openai/gpt-oss-120b`, unchanged). All three cost-rate variables **absent** |
| Embedding provenance | **[REPRODUCED FACT]** `model.safetensors` sha256 matches `MODEL_PROVENANCE.json` exactly |

**Test suite.** The pre-audit suite was 23 tests, which reproduces. But
**[REPRODUCED FACT]** `pytest` was **not installed in `.venv`** — the documented
command ran under `C:\ProgramData\anaconda3\python.exe`, a different environment
carrying `scipy`, `pandas`, `torchvision` and `pytest 7.4.4`. The suite never
validated the environment that runs the research. Fixed; the suite now runs in
`.venv` and stands at **73 passing** (`TEST_REPORT.txt`).

### Test coverage was materially weaker than the count suggested

**[REPRODUCED FACT]** The pre-audit 23 tests completed in 1.5 s, too fast to
touch the real encoder. Specifically:

- `test_probe_depth_is_not_final_k_and_dedup_is_after_retrieval` and
  `test_context_budget_is_never_silently_exceeded` assert against the **legacy**
  `LlmJudgedSarRagSystem`, not the primary `SarRagSystem`. The two headline
  contracts were tested on the wrong system.
- The single primary-pipeline test set `similarity_threshold=-1.0`,
  `novelty_threshold=0.0`, `marginal_utility_threshold=0.0` — **every adaptive
  gate disabled**.
- Nothing covered `relevance_percentile`, `score_scale`, the NRRF normalization,
  the frozen manifest actually being applied, the real MiniLM encoder, or any
  harness contract.

I added **50 tests** in three files: `test_primary_pipeline_contracts.py` (34),
`test_experiment_contracts.py` (10), `test_defect_regressions.py` (6).

---

## 3. Phase 2 — Code-to-protocol audit

Full traceability matrix: **`CODE_PROTOCOL_TRACEABILITY.csv`** (47 rows: intended
step, file, function, actual behaviour, evidence, mismatch, severity, proposed
correction).

Verdicts on the specifically required checks, all **[REPRODUCED FACT]** and now
test-pinned:

| Required check | Verdict | Evidence |
|---|---|---|
| Probe top-3 separate from final k | **HOLDS** | probe emits 12 candidates; `final_k` varies with budget and never equals 3 |
| All four views probed | **HOLDS** | `_probe` iterates all enabled views |
| RRF selects representations | **HOLDS** | deterministic, no LLM reachable from `selection.py` |
| At most two views selected | **HOLDS** | enforced in config and selector |
| Adaptive retrieval uses only selected views | **HOLDS** | every scored candidate's representation ⊆ selected |
| Final k not fixed | **HOLDS** | emergent; does not sit at the cap under a generous budget |
| Dedup after retrieval | **HOLDS** | accepted-candidate IDs == provisional-evidence IDs, so the probe removed nothing |
| Exactly one primary LLM call | **HOLDS** | 1 `answerer.generate`; `api_calls == 1.0` per arm per record |
| Traditional and SAR answerers identical | **HOLDS** | both construct `GroqOneCallAnswerGenerator`; decoding controls set once in `GroqClient.complete` |
| Traces contain required events | **HOLDS** | stop reason, rejection reason, rounds, pause reasons all present |
| Safety caps cannot be silently exceeded | **HOLDS** | token, chunk, candidate and round caps each verified |

### Deviations from the written protocol

**D08 — the RRF formula in the paper is wrong. [REPRODUCED FACT]** The protocol
states `NRRF(g) = RRF(g) / (n_representations/(c+1))`. The selector computes
that, **then divides each representation's NRRF mass by the summed NRRF mass of
all aligned probe groups** (`selection.py:81-82`). The frozen **0.2437 is a
share of total aligned probe mass**, not a raw NRRF quantity. The definition is
consistent between calibration and inference, so the threshold remains valid —
but publishing the current formula would misstate the method.

**D09 — the calibration selection rule has an undocumented second filter.**
Both readings are stated below in §5; neither was silently changed.

**D11/D12 —** tie-break uses probe *words* not *tokens*; prose describes
patience 2 while the frozen value is 3.

---

## 4. Phase 3 — Dataset and leakage audit

Full artifact: **`DATASET_LEAKAGE_AUDIT.json`**.

**[REPRODUCED FACT]** Split counts reproduce exactly:

| Split | Questions | Unique IDs | Answerable | Insufficient |
|---|---:|---:|---:|---:|
| QASPER test | 1,451 | 1,451 | 1,372 | 79 |
| MuSiQue dev | 4,834 | 4,834 | 2,417 | 2,417 |

Zero duplicate example IDs in either file. MuSiQue answerable/insufficient
variants are perfectly paired, confirming correction #2 (`:answerable` /
`:insufficient` / `:hidden` suffixes).

### The 72 QASPER overlaps, characterized

**[REPRODUCED FACT]** 72 distinct question hashes overlap train↔test, touching
**163 test examples (11.2%)**, with **0** paper-ID and **0** document-ID overlap.
I inspected all 72. Every one is a generic NLP-paper template — *"What was the
baseline?"* (9 test examples), *"Do they report results only on English data?"*
(9), *"What datasets did they use?"* (4), and so on — and **zero share a paper
with their training counterpart**.

**[REVIEWER INFERENCE]** This is question-template repetition, not answer or
source leakage: the answer to *"What was the baseline?"* is determined entirely
by the paper, and no paper is shared. It must still be disclosed, and must
**not** be called zero leakage. It is intrinsic to QASPER — train contains 158
internally duplicated question strings and test 108.

### The check that mattered and was missing

**[REPRODUCED FACT]** Only the 60 calibration examples were ever fitted. Against
that sample:

| | QASPER test | MuSiQue dev |
|---|---:|---:|
| Example-ID overlap | 0 | 0 |
| Source/paper-ID overlap | 0 | 0 |
| Document-ID overlap | 0 | 0 |
| Distinct question templates shared | 1 | 0 |
| Evaluation examples touched | 2 (0.14%) | 0 (0.0%) |

Calibration fits 8 scalar thresholds plus one baseline *k* using gold evidence
spans and answerability strata only — no answer text, no Groq call. **[REVIEWER
INFERENCE]** The exploitable channel from template repetition is very small.

**[LIMITATION]** Exact matching cannot rule out semantic paraphrase, prior
benchmark contamination, or `openai/gpt-oss-120b` pretraining exposure. The last
applies **equally to both arms**, so it does not bias the paired comparison —
only absolute scores.

---

## 5. Phase 4 — Calibration reproduction

**Existing artifacts were preserved.** `output/calibration/frozen_manifest.json`
was never modified. New outputs went to `output/independent_verification/`.

**[REPRODUCED FACT]** Verified against the manifest:

| Claim | Verified |
|---|---|
| Status `training_only_calibration_frozen` | yes |
| 60 examples, 12 strata, 5 per stratum | yes (30 QASPER + 30 MuSiQue) |
| 32 proposed configurations | yes |
| 6 baseline-k candidates | yes — {3, 5, 8, 10, 15, 20} |
| No Groq calls, no gold answer text | yes — `RetrievalOnlyAnswerer`, evidence spans only |
| Frozen controls | all 8 match the brief exactly; `static_k = 20` |
| Source hashes | match |

**Three-way reproduction result — every compared field identical:**

| Comparison | Result |
|---|---|
| v0 original frozen ↔ v1 reproduction (unmodified code) | **IDENTICAL** across all 17 compared fields |
| v1 reproduction ↔ v2 (after defect fixes) | **IDENTICAL** across all 17 compared fields |
| Thresholds changed by remediation | **none** |

The v2 manifest adds only `corpus_cache_policy`, recording `corpora_built: 60`
against 58 unique corpora — the two predicted rebuilds from the one-active-corpus
refactor. Results are unaffected because each (example, configuration)
evaluation is independent given the corpus.

**[REPRODUCED FACT] Held-back calibration diagnostics reproduce exactly:**

| Metric | SAR-RAG | Traditional |
|---|---:|---:|
| Evidence recall | 0.7917 | 0.8958 |
| Complete-evidence recovery | 0.6875 | 0.8750 |
| Evidence precision | 0.1719 | 0.1127 |
| Mean context tokens | 1,265 | 1,833 |
| Mean final k | 14.67 | 17.33 |
| Mean selected representations | **1.125** | 1.0 |

**[CALIBRATION DIAGNOSTIC]** These are calibration diagnostics on 24 held-back
training examples. They do not establish superiority. They show SAR-RAG more
evidence-precise and using ~31% less context, but recovering **less** total and
complete gold evidence.

### D09 — the two selection rules, stated side by side

**Documented rule** (`CALIBRATION_AUDIT.md`, and the brief): shortlist
configurations within 0.01 of best **tuning** gold-evidence recall; then on the
held-back partition maximize complete-evidence recovery → evidence precision →
minimize context tokens → minimize candidates examined → serialized parameter
order.

**Implemented rule** (`calibration.py:176-178`): identical, **except** that
`choose_configuration` re-filters the shortlist to configurations within 0.01 of
the best **selection-partition** recall before applying the lexicographic order.

**Recommendation: adopt the implemented rule as the version-2 documented rule.**
**[REVIEWER INFERENCE]** Justification: the extra filter is strictly more
recall-protective. Removing it would allow a configuration with materially lower
held-back recall to win on a complete-evidence tie, which contradicts the stated
priority that evidence coverage outranks efficiency. It did not bias selection
toward the efficiency-favouring configuration. And correcting prose is far safer
than editing selection code after results exist.

### D10 — the partition is 60/40, not 70/30

**[REPRODUCED FACT]** `round(5 × 0.30) = 2` under banker's rounding, so each
5-example stratum yields 3 tuning + 2 selection → **36 tuning / 24 selection**.
Documentation corrected for version 1. A clean 70/30 requires a preregistered
per-stratum size that divides evenly — **10 per stratum → 7/3**, giving a
120-example sample. That must be preregistered, not retrofitted.

---

## 6. Phase 5 — Metric verification

**[REPRODUCED FACT]** I recomputed every metric by hand and compared against the
implementation. **22 of 22 checks pass.**

- **Normalization**: `"The Policy, applies. [E1]"` → `"policy applies"` —
  confirms `[E#]` removal, case folding, punctuation removal, article removal,
  whitespace collapsing.
- **Token PRF**: multiset overlap; best-F1 gold selection verified.
- **Retrieval**: `evidence_set_recall`, `evidence_set_precision`,
  `complete_evidence_recovery`, MRR, P@k and nDCG@k all match hand computation.
  All metrics correctly return **NaN** with no gold denominator.
- **Exact McNemar**: two-sided exact binomial verified on two cases (p = 1.0 and
  p = 0.0625).
- **Holm**: step-down with monotonicity enforcement verified.

### Metric issues that remain open

**D17 [LIMITATION]** `precision_at_k` divides by the **cutoff**, not by the
number retrieved, so an arm with smaller emergent `final_k` is penalised at
large *k*. Combined with the granularity difference, chunk-level P@k, MRR and
nDCG are **not comparable across arms** and must stay strictly secondary. The
module already carries this caveat; the paper must repeat it.

**D14 [REPRODUCED FACT] — complete-evidence recovery is scored on unanswerable
questions.** `iter_musique_full` collects `is_supporting` paragraphs regardless
of answerability. Consequently:

| Split | No gold spans (metrics NaN) | Insufficient **with** gold spans |
|---|---:|---:|
| QASPER test | 99 (79 insufficient + 20 answerable) | 0 |
| MuSiQue dev | 541 | **1,876** |

So 1,876 MuSiQue examples (39% of that set) are declared unanswerable yet scored
for recall and complete-evidence recovery. **[REVIEWER INFERENCE]** Retrieving
all residual supporting paragraphs for a question that is unanswerable *because a
hop is missing* is not obviously the right target; pooling these with answerable
items makes complete-evidence recovery hard to interpret. **This requires a
version-2 decision before any headline CER number is published.**

**D15 [REPRODUCED FACT] — the four primary outcomes have very different valid
paired n:**

| Primary outcome | Valid paired n (of 6,285) |
|---|---:|
| `answer_token_f1` | **3,789** |
| `evidence_set_recall` | 5,645 |
| `complete_evidence_recovery` | 5,645 |
| `context_tokens` | 6,285 |

Holm correction across outcomes with different *n* is legitimate, but reporting
a single "n = 6,285" beside all four would be misleading.

**Cost.** **[PENDING]** All three rate variables are absent and
`experiments.py` correctly refuses confirmatory mode without them. Per the audit
rules, token usage is reported and **cost is left undefined**. See
`EXACT_PAID_EXECUTION_PLAN.md` §3 for the formula and measured token volumes.

---

## 7. Phase 6 — Statistical plan review

**[REPRODUCED FACT]** The harness implements exactly what was specified: paired
bootstrap 95% CI, paired randomization p-value, paired `dz`, valid paired *n*,
exact McNemar on the three binary outcomes, and Holm over the four primary
outcomes. Resampling is memory-bounded (batched to ≈2M temporaries) rather than
allocating `resamples × examples`.

### Are the predeclared margins defensible?

**[REVIEWER INFERENCE]**

| Margin | Assessment |
|---|---|
| Answer token F1 superiority ≥ +0.02 | **Defensible.** Small but conventional for QA F1. |
| Evidence recall non-inferiority, lower bound > −0.02 | **Not currently defensible.** Calibration shows −0.104 — five times the margin. Declaring non-inferiority at −0.02 while the only evidence available shows −0.104 sets up a test the method is very likely to fail. That is *good* preregistration, but the team must accept the likely outcome rather than widen the margin afterwards. |
| Complete-evidence non-inferiority > −0.02 | **Same concern, worse.** Calibration shows −0.187. |
| Context-token reduction ≥ 10% | **Defensible as an efficiency target, but not yet attributable.** Calibration shows 31%, and the granularity confound (D13) means an unknown share is chunk size, not adaptivity. |
| Cost reduction ≥ 10% | **Weak as stated.** Cost is dominated by prompt tokens, so this is largely a restatement of the context-token margin, not independent evidence. |

**These margins must not be changed after held-out results are seen.** I am
recording the concern *now*, before any confirmatory run, precisely so that a
later adjustment would be visible as post-hoc.

---

## 8. Phases 7–13 — Execution status

| Phase | Status |
|---|---|
| 7 — Full held-out evaluation | **[PENDING]** — not authorized; 0 paid calls made |
| 8 — Representation-selection analysis | **retrieval-only evidence produced** on training data |
| 9 — Adaptive-k verification | **[PENDING]** end-to-end; retrieval-only `final_k` distributions produced |
| 10 — Registered ablations | **15 of 20 screened offline**; staged plan in `EXACT_PAID_EXECUTION_PLAN.md` §8 |
| 11 — Robustness/subgroup | **[PENDING]** — requires per-question held-out records |
| 12 — Failure analysis | **[PENDING]** — the analyzer needs fields the stale pilot lacks |
| 13 — Q2 analysis | **[PENDING]** — no confirmatory results to analyze |

**[REPRODUCED FACT]** `table_failures.csv` and `manual_failure_review_sample.csv`
were missing because `_write_csv` silently returned on empty input **and** the
stale pilot lacks `gold_evidence_spans`, `provisional_evidence` and
`representation_gold_diagnostic`, so `failure_analysis` produced no rows. Both
causes are fixed (D05); populated failure tables still require a run by the
current code.

Only 5 of the 9 required figures exist (`01`–`04`, `06`; `05` absent), and all
were built from pilot data. **They must not be relabelled as final.**

---

## 8b. Retrieval-only ablation screening — the most important empirical section

**Method.** `tools/retrieval_only_screening.py`, **zero Groq calls**, on a fresh
deterministic stratified sample of **60 official TRAINING examples**
(seed `20260804`, deliberately different from the calibration seed `20260803` so
screening does not re-use the examples that selected the thresholds). No
evaluation split was touched. 43 of 60 carry gold evidence and are scored.

**[REPRODUCED FACT] SAR-RAG conditions** (frozen controls, retrieval-only):

| Condition | Recall | Complete-ev. | Precision | Ctx tokens | final k | Cand. | 2-view % |
|---|---:|---:|---:|---:|---:|---:|---:|
| **full** | 0.7054 | 0.5814 | 0.1595 | 1312.7 | 15.15 | 33.4 | 16.7 |
| **paragraph_only** | **0.8004** | **0.6512** | **0.1751** | **1281.2** | 15.27 | **22.9** | 0.0 |
| sentence_window_only | 0.6996 | 0.5349 | 0.2043 | 1321.3 | 15.77 | 22.6 | 0.0 |
| semantic_only | 0.7171 | 0.5581 | 0.1492 | 1358.7 | 16.43 | 24.2 | 0.0 |
| hierarchical_only | 0.3876 | 0.2326 | 0.0722 | 1408.9 | 14.18 | 23.8 | 0.0 |
| all_representations_no_rrf | 0.5950 | 0.3953 | 0.1375 | 1056.2 | 11.00 | 38.6 | 0.0 |
| rrf_probe_only_no_adaptive | 0.4671 | 0.3256 | 0.3566 | 303.5 | 3.33 | 12.0 | 16.7 |
| no_novelty | 0.7054 | 0.5814 | 0.1595 | 1312.7 | 15.15 | 33.4 | 16.7 |
| no_similarity_gap | 0.7054 | 0.5814 | 0.1595 | 1312.7 | 15.15 | 33.4 | 16.7 |
| no_post_retrieval_dedup | 0.7054 | 0.5814 | 0.1602 | 1378.5 | 16.10 | 33.4 | 16.7 |
| select_one_representation | 0.6938 | 0.5814 | 0.1598 | 1329.0 | 15.40 | 32.3 | 0.0 |
| probe_depth_5 | 0.7519 | 0.6279 | 0.1577 | 1330.7 | 14.80 | 39.0 | 10.0 |
| rrf_k0_1 / _10 / _60 | 0.7054 / 0.7171 / 0.7171 | 0.5814 | ~0.161 | ~1305 | ~15.1 | 33.3 | 16.7–18.3 |

**[REPRODUCED FACT] Traditional baseline — granularity × fixed k:**

| Condition | Mean words | Recall | Complete-ev. | Precision | Ctx tokens | final k |
|---|---:|---:|---:|---:|---:|---:|
| sliding_window k=3 | 98.4 | 0.4535 | 0.2791 | 0.3256 | 469.2 | 3.00 |
| sliding_window k=5 | 98.4 | 0.5504 | 0.3721 | 0.2419 | 760.9 | 5.00 |
| **sliding_window k=10** | 98.4 | **0.6492** | **0.4651** | 0.1628 | **1293.0** | 9.67 |
| sliding_window k=20 | 98.4 | 0.7733 | 0.7209 | 0.1209 | 1896.4 | 15.78 |
| sliding_window_matched k=10 | 67.8 | 0.6240 | 0.4651 | 0.2023 | 953.7 | 10.00 |
| sliding_window_matched k=20 | 67.8 | 0.7674 | 0.6744 | 0.1392 | 1794.8 | 19.13 |

### Five findings that change the assessment

**F1 — A single-view baseline beats the full routed system (D21). [REPRODUCED
FACT]** `paragraph_only` beats `full` on **every** retrieval outcome
simultaneously: recall +0.095, complete-evidence +0.070, precision +0.016,
**fewer** context tokens, and 31% fewer candidates examined. Always using the
paragraph view is simpler, cheaper and better on this training sample. **This is
the most serious threat to the paper's contribution.** The routing mechanism must
be justified against *this* null, not only against `all_representations_no_rrf`.

**F2 — Two named components are inert (D22). [REPRODUCED FACT]** `no_novelty`
and `no_similarity_gap` reproduce `full` to four decimal places on every metric.
At the frozen thresholds (novelty 0.2292, standardized gap 1.9052) **neither rule
ever binds** on this sample. They cannot be claimed as contributing components.

**F3 — At a matched context budget, SAR-RAG does win. [REPRODUCED FACT]** This
is the one clearly favourable result. Comparing at ~1,300 tokens:

| At ≈1,300 context tokens | Recall | Complete-evidence |
|---|---:|---:|
| SAR-RAG `full` (1,313 tok) | **0.7054** | **0.5814** |
| Traditional k=10 (1,293 tok) | 0.6492 | 0.4651 |
| Difference | **+0.056** | **+0.116** |

**[REVIEWER INFERENCE]** The correct framing of the efficiency claim is
**budget-matched, not k-matched**. Against k=20 (1,896 tokens) SAR-RAG is worse
on coverage; against an equal token budget it is better. The paper should make
the budget-matched comparison primary, and must present both.

**F4 — I overstated the granularity confound; correcting myself (D13).
[REPRODUCED FACT]** My earlier HIGH severity was inferred from configuration
parameters (256 vs 95 words = 2.7×) without measuring realized output. Measured:

| Dataset | Mean adaptive view | Baseline sliding window | Ratio |
|---|---:|---:|---:|
| QASPER | 63.3 words | 109.5 words | **1.73×** |
| MuSiQue | 54.9 words | 79.0 words | **1.44×** |

Sliding windows terminate early at parent-section boundaries, so the realized gap
is roughly half what the parameters imply. Severity revised **HIGH → MEDIUM**.
The confound is real and must be disclosed, but it does **not** explain away the
matched-budget result in F3.

**F5 — The second representation contributes almost nothing, and only on QASPER
(D24). [REPRODUCED FACT]**

| Diagnostic | Value |
|---|---|
| One view selected | 83.3% |
| Two views selected | 16.7% (10/60) |
| Two-view by dataset | QASPER 9/30 (30%) vs MuSiQue **1/30 (3.3%)** |
| Second view added unique probe gold | 20% of two-view cases = **3.3% of all queries** |
| Unselected view holds unique gold (routing risk) | 6.98% of scored examples |
| `select_one_representation` cost | −0.0116 recall, **0.000** complete-evidence |

Most-selected first views: paragraph 34, semantic 16, sentence_window 7,
hierarchical 3. **[REVIEWER INFERENCE]** The "one *or two* complementary
representations" framing is not supported as a general mechanism; it is
essentially a QASPER-only effect worth ~0.01 recall.

### Supporting findings

- **Adaptive retrieval does real work.** `rrf_probe_only_no_adaptive` reaches
  only 0.4671 recall at 303 tokens; adaptive retrieval adds **+0.238 recall** for
  4× the context. This component *is* supported.
- **RRF selection beats no selection.** `all_representations_no_rrf` scores
  0.5950 vs `full` 0.7054, so selecting is better than using all four views —
  but still worse than simply always using paragraph (F1).
- **The RRF constant barely matters (D23).** `rrf_k0 ∈ {1,3,10,60}` moves recall
  only 0.7054→0.7171 with complete-evidence unchanged. Report as robustness, not
  as a tuned advantage.
- **Deduplication is mildly positive.** Removes ~0.95 chunks and 66 tokens with
  **zero** recall loss.
- **Probe depth 3 is not optimal (D25).** `probe_depth_5` gives +0.047 recall and
  +0.047 complete-evidence. **This must not be silently adopted** — it would be
  post-hoc tuning. It requires a fresh training-only calibration if adopted.
- **36.7% of queries stop on a safety cap (D26).** `final_chunk_safety_cap`
  30.0%, `retrieval_round_safety_cap` 6.7%, adaptive pause 63.3%. Mean final k
  15.15 of a 20 cap. The system is not purely a capped retriever, but this
  materially weakens the "no fixed k" framing and must be disclosed.

**[LIMITATION]** All of §8b is **training-data, retrieval-only** evidence on 60
examples (43 scored). It carries no answer-quality, latency or cost information,
and no statistical inference is attached. It is screening evidence for deciding
what deserves paid evaluation — not a result.

---

## 9. Required final review decision

| # | Question | Answer |
|---|---|---|
| 1 | Does the implementation match the intended pipeline? | **Substantially yes**, with one formula misstatement (D08) and documented deviations D09–D12. Every structural contract in Phase 2 holds and is now test-pinned. |
| 2 | Does SAR-RAG use exactly one primary LLM call? | **Yes.** **[REPRODUCED FACT]** Verified structurally and end to end (`api_calls == 1.0` per arm per record). A paired study with the evaluator is **3 calls per question**; this must never be conflated with the method's 1. |
| 3 | Is final k genuinely adaptive? | **Yes in mechanism** — emergent, varies with budget, does not sit at the cap. **[PENDING]** whether it is *meaningfully* adaptive at scale. |
| 4 | Is representation selection truly based on RRF complementarity? | **Yes in mechanism**, but the published formula is wrong (D08), and the second view is selected in ~12.5% of calibration cases (D16). |
| 5 | Does deduplication happen after retrieval? | **Yes.** **[REPRODUCED FACT]** Accepted-candidate IDs equal provisional-evidence IDs, proving the probe removed nothing. |
| 6 | Is Traditional RAG a fair baseline? | **Fair on the answer side** — identical generator, prompt, schema, model, decoding. **Not yet fair to interpret on efficiency** because of the granularity confound (D13). |
| 7 | Was calibration isolated from evaluation? | **Yes.** **[REPRODUCED FACT]** Training-only sources, no Groq, no answer text; confirmatory mode requires a frozen manifest and rejects `--limit` and baseline-k overrides. |
| 8 | Evidence of leakage or contamination? | **No source/document/example leakage.** Question-template repetition exists and is disclosed; the calibration-level channel is 0.14% of QASPER test and 0% of MuSiQue. |
| 9 | Are all metrics correctly calculated? | **Yes, 22/22 against hand computation** — with the granularity caveat on rank metrics (D17) and the unresolved CER-on-unanswerable question (D14). |
| 10 | Are tests paired and appropriately corrected? | **Yes** — paired bootstrap, randomization, exact McNemar, Holm over four outcomes. Valid *n* differs per outcome and must be reported (D15). |
| 11 | Are observed changes practically meaningful? | **[PENDING]** — no held-out data. Calibration suggests a real efficiency gain and a real evidence-coverage loss. |
| 12 | Does SAR-RAG improve answer quality? | **[PENDING]** — unknown. Not demonstrated anywhere. |
| 13 | Does SAR-RAG improve evidence recall? | **No, on current evidence.** −0.104 on held-back calibration. |
| 14 | Does SAR-RAG improve complete-evidence recovery? | **No, on current evidence.** −0.187 on held-back calibration. |
| 15 | Does it improve precision and context efficiency? | **Yes**, and this is now better supported than before: at a matched ~1,300-token budget SAR-RAG beats Traditional k=10 by +0.056 recall and +0.116 complete-evidence (§8b F3). Granularity contributes (1.44–1.73×) but does not explain it. |
| 16 | Is it a meaningful Pareto improvement? | **Against fixed-k Traditional at a matched budget, plausibly yes.** Against a `paragraph_only` single-view baseline, **no** — that null dominates SAR-RAG on every retrieval outcome at lower cost (§8b F1). |
| 17 | Which components are supported by ablations? | **Supported:** adaptive retrieval (+0.238 recall over probe-only), RRF selection over no selection (+0.110), post-retrieval dedup (mildly). **Not supported:** novelty and similarity-gap are provably inert (F2); the second representation is worth ~0.01 recall on 3.3% of queries (F5); the RRF constant is irrelevant (D23); the hierarchical view is by far the weakest (0.3876 alone). **[LIMITATION]** retrieval-only, training data. |
| 18 | What are the dominant failure modes? | **[PENDING]** end-to-end. Screening identifies two structural ones: 36.7% of queries terminate on a safety cap rather than adaptively (D26), and an unselected view holds unique gold evidence in 7.0% of scored queries. |
| 19 | Strong enough for Q2 submission? | **Not yet.** No confirmatory result exists. The engineering is sound and now well-tested, but the empirical case is unbuilt. |
| 20 | What claims are supported / unsupported / must be weakened? | See §10. |

---

## 10. Claim ledger

**Supported now [REPRODUCED FACT]:**
- SAR-RAG is model-fine-tuning-free and training-calibrated. No embedding or LLM
  weights are trained; 8 thresholds + 1 baseline *k* are fitted on 60 training
  examples.
- The primary method issues exactly one online LLM call per question.
- Representation selection, probing, RRF, stopping, novelty and deduplication
  are LLM-free.
- Deduplication occurs strictly after adaptive retrieval.
- Final *k* is emergent, not configured.
- The calibration is deterministic and reproducible bit-for-bit.
- Calibration is isolated from evaluation; no source/document/example leakage.
- Both arms use an identical answer generator, so only retrieval differs.

**Unsupported — no evidence exists [PENDING]:**
- Any statement that SAR-RAG outperforms Traditional RAG on answer quality.
- Any held-out evidence recall, complete-evidence recovery, answerability,
  latency, cost or final-*k* distribution.
- Any claim that all four representations are necessary.
- Any claim about failure modes or robustness.

**Must be weakened or corrected before submission:**
1. **The RRF formula must be rewritten** (D08): 0.2437 is a fraction of total
   aligned probe mass, after a second normalization the paper does not describe.
2. **"Training-free" must never appear unqualified** — the repository is already
   correct on this; keep it.
3. **Never call normalized RRF a calibrated probability.**
4. **Do not describe QASPER leakage as zero.** State: 72 repeated generic
   question templates, 163 test examples, 0 shared papers, and a 0.14%
   calibration-level channel.
5. **Report the efficiency comparison budget-matched, not k-matched** (§8b F3),
   and disclose the measured 1.44–1.73× granularity difference (D13).
6. **Do not present "one or two representations" as a general mechanism.** Two
   views are selected for 16.7% of queries, add unique gold evidence in 3.3%,
   and the effect is essentially QASPER-only (D24).
11. **Do not claim novelty or the similarity-gap rule as contributing
    components.** Both are provably inert at the frozen thresholds (D22).
12. **Do not claim the four-view design is necessary.** `paragraph_only`
    currently dominates the full system, and `hierarchical_only` is by far the
    weakest view (D21).
13. **Do not describe final k as unconstrained** while 36.7% of queries stop on a
    safety cap (D26).
14. **Do not adopt probe depth 5** on the basis of screening; it requires a fresh
    training-only calibration (D25).
7. **Do not report a single n = 6,285** beside all four primary outcomes (D15).
8. **Do not report chunk-level P@k / MRR / nDCG as primary** (D17).
9. **Do not describe the three-calls-per-question study cost as the method's
   cost.**
10. **Do not label pilot figures as final results.**

---

## 11. Current defensible conclusion

**[REVIEWER INFERENCE]** My reproduced evidence does not change the stated
paper-safe conclusion, and I endorse it with one addition:

> SAR-RAG is a model-fine-tuning-free, training-calibrated adaptive retrieval
> method. Training calibration indicates higher evidence precision and reduced
> context use, but lower evidence recall and complete-evidence recovery than a
> static-*k* Traditional RAG baseline. Its overall quality–efficiency advantage
> remains unproven pending complete held-out evaluation, registered ablations,
> systems measurements and manual failure analysis.

**Three auditor additions, all from retrieval-only screening on training data:**

1. **The efficiency comparison must be budget-matched, not k-matched.** At an
   equal ~1,300-token budget SAR-RAG beats Traditional (+0.056 recall, +0.116
   complete-evidence). Against k=20 it loses on coverage. Both must be reported;
   the budget-matched framing is the honest one and is the strongest result the
   method currently has.
2. **Two named components are inert and one is near-inert.** Novelty and the
   similarity-gap rule change nothing at the frozen thresholds; the second
   representation is worth ~0.01 recall on 3.3% of queries. These cannot be
   presented as contributing mechanisms.
3. **A trivial single-view baseline currently dominates the method.**
   `paragraph_only` beats the full routed system on recall, complete-evidence,
   precision, context tokens and candidates examined simultaneously. **Until this
   is refuted on held-out data, the central contribution is not established.**

**Outcome classification: (D) no convincing advantage — with a genuine but
narrow (C) efficiency-oriented trade-off at matched context budget.** Neither can
be asserted as a *result*, because no confirmatory experiment has been run. The
experiment is **not** invalid (E): no leakage, unfair prompt, broken metric or
test-set tuning was found. It is *unexecuted* — and the screening evidence now
predicts that the routing contribution, as currently framed, will not survive it.

---

## 12. Unresolved issues

| # | Issue | Blocking? |
|---|---|---|
| U1 | No confirmatory run exists; all held-out results are **[PENDING]** | **Yes** — for any empirical claim |
| U2 | Groq run-date rates unverified; cost undefined | **Yes** — for any cost claim |
| U3 | D14: CER scored on 1,876 unanswerable MuSiQue examples — needs a version-2 decision | **Yes** — for any CER headline |
| U4 | D13: granularity confound — control added, not yet reported end to end | **Yes** — for any efficiency attribution |
| U5 | D08: paper RRF formula does not match the implementation | **Yes** — for method description |
| U6 | D16/D24: second view selected in 16.7% of queries, adds unique gold in 3.3%, essentially QASPER-only | **Yes** — for the routing claim |
| U13 | **D21: `paragraph_only` dominates the full routed system on every retrieval outcome at lower cost** | **Yes** — this is the central threat to the contribution |
| U14 | D22: novelty and similarity-gap components are provably inert at the frozen thresholds | **Yes** — for any component claim |
| U15 | D26: 36.7% of queries terminate on a safety cap rather than adaptively | **Yes** — for the "no fixed k" framing |
| U16 | D25: probe depth 5 outperforms the frozen depth 3; must not be adopted without fresh calibration | No — but do not silently tune |
| U7 | Non-inferiority margins likely unachievable on current evidence | No, but must not be revised post-hoc |
| U8 | Failure tables and 4 of 9 figures cannot be produced without a real run | No |
| U9 | D19: confirmatory figure hard-codes a "pilot" caption | No — cosmetic, fix in v2 |
| U10 | D20: revision pin inert on local snapshot path; provenance rests on sha256 | No — hash verified |
| U11 | `pytest` still not declared in `pyproject.toml` | No — add a test extra |
| U12 | MuSiQue test labels hidden; test-set numbers require official submission | No — dev used for local scoring |
