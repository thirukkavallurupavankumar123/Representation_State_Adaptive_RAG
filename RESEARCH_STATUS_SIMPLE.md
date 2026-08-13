# Representation-State Adaptive RAG

## 1. Final research definition

This research studies whether a RAG system can choose the useful document chunking views for each question and stop retrieval dynamically, instead of always using one chunking method and a fixed final k.

The method is called Representation-State Adaptive RAG, or SAR-RAG. It is model-fine-tuning-free, but it is not supervision-free. Gold evidence from official training splits is used to calibrate mathematical thresholds. No embedding or language-model weights are trained.

The primary online method uses exactly one LLM call per question. That call happens only after retrieval and deduplication, to produce the final grounded answer or abstention. Retrieval evaluation may use a separate blinded evaluator call; that call is part of the experiment, not part of SAR-RAG.

## 2. Research question

Does selecting one or two useful representations from a small mathematical probe, followed by adaptive retrieval and post-retrieval deduplication, provide a better evidence-quality and efficiency trade-off than Traditional dense RAG using one sliding-window index and a training-calibrated static k?

The study must not simply claim that SAR-RAG wins. It must measure answer quality, evidence recovery, faithfulness, latency, tokens, cost, and offline indexing cost. A method is not better overall if it saves context but misses necessary evidence.

## 3. Fixed models and data

- Embedding model: sentence-transformers/all-MiniLM-L6-v2, loaded with Hugging Face Transformers and PyTorch.
- Embedding revision: 1110a243fdf4706b3f48f1d95db1a4f5529b4d41.
- Pooling: attention-mask mean pooling followed by L2 normalization.
- Maximum embedding length: 256 tokenizer tokens.
- Final answer model: Groq openai/gpt-oss-120b, unless the environment records a different frozen model before evaluation.
- Primary datasets: QASPER v0.3 and MuSiQue-Full v1.0.
- QASPER purpose: long scientific documents and evidence at different granularities.
- MuSiQue purpose: connected multi-hop questions and paired insufficient-evidence cases.
- HotpotQA: optional external generalization only.
- QuALITY: optional future context-granularity experiment.
- Custom representation-state benchmark: useful only after it is authored, audited, and split before testing. It is not an existing downloadable public dataset.

## 4. End-to-end online algorithm

### Step 1 - Read and structure the source

The system accepts one or more documents. It parses them into canonical source units with stable IDs, document and version IDs, section paths, order, and text offsets.

Why this exists: every representation must point back to the same original units. Without this shared source map, overlap, deduplication, citations, and evidence evaluation become unreliable.

Better alternative when available: use layout-aware PDF parsing for visually complex PDFs. The current primary experiment should use digitally extractable text or manually verify extraction quality.

### Step 2 - Build four representations

- Sentence window: a sentence with nearby context.
- Paragraph: a bounded paragraph passage.
- Semantic segment: adjacent units grouped by semantic continuity.
- Hierarchical section: bounded text with its parent heading.

All non-window groups are partitioned to about 95 words before embedding. Every chunk records its original tokenizer length and whether truncation occurred. A real QASPER audit produced 343 chunks, zero truncations, and a maximum of 231 MiniLM tokens.

Why this exists: different questions need different granularity. A precise fact may favor a sentence or paragraph; a scoped rule may need its section heading; a connected explanation may need a semantic segment.

Better alternative: learn a query router, but that adds training, a new model, and another possible source of bias. The primary method deliberately tests a mathematical router.

### Step 3 - Embed and index each representation

Each representation receives a separate exact cosine index using the same MiniLM encoder. Exact search is used in the controlled experiment so approximate-nearest-neighbor settings do not confound retrieval quality.

Why this exists: separate indexes allow representation-specific ranking. Exact search makes the experiment reproducible.

Better alternative for deployment: an approximate index such as HNSW after the exact-search research result is established. Its recall and latency must be reported separately.

### Step 4 - Probe top 3 from every representation

For a new question, retrieve ranks 1 to 3 from all four indexes. Top 3 is the probe depth, not the final retrieval k. No LLM is called.

Why this exists: the probe cheaply reveals which views surface related source units. A deeper probe costs more and begins to behave like full retrieval.

Alternative: probe depth 5 is a predeclared sensitivity experiment, not a value chosen after looking at test results.

### Step 5 - Align probe results and calculate RRF

Probe chunks are grouped when they point to equal, contained, or strongly overlapping canonical units. For evidence group g:

RRF(g) = sum across representations retrieving g of 1 / (c + rank)

Normalized RRF(g) = RRF(g) / (number of representations / (c + 1))

The frozen value is c = 3. A small constant is appropriate because only ranks 1 to 3 are probed; c = 60 would make those ranks nearly equal.

The first representation is the one covering the largest total normalized-RRF mass. A second representation is selected only if it covers enough RRF mass not already covered by the first. The frozen marginal-gain threshold is 0.2437. Ties use lower mean probe-token cost and then stable representation name.

Why this exists: agreement increases confidence, while uncovered RRF mass measures complementarity. The formula selects representations, not individual final chunks.

Better alternative: weighted RRF could learn per-view reliability, but it requires additional labeled training and can overfit. Uniform RRF is the cleaner primary condition.

### Step 6 - Retrieve adaptively with no fixed final k

Reuse useful selected probe chunks. Then read further candidates from the selected one or two indexes in batches of three.

For each candidate:

- Relevance is its empirical cosine-score percentile inside its own representation index.
- Canonical overlap is the largest source-unit Jaccard overlap with accepted evidence.
- Semantic similarity is the largest embedding cosine with accepted evidence.
- Novelty = 1 - canonical overlap multiplied by the positive part of semantic similarity.
- Marginal utility = relevance multiplied by novelty.

Raw cosine thresholds are not compared across views because score distributions differ. A score percentile is used instead. Score gaps are divided by that index's score standard deviation.

The frozen controls are:

- Relevance percentile threshold: 0.5289.
- Standardized similarity-gap threshold: 1.9052.
- Novelty threshold: 0.2292.
- Redundancy threshold: 0.9547.
- Marginal-utility threshold: 0.1289.
- Empty-batch patience: 3.
- Maximum selected representations: 2.
- Maximum final chunks: 20.
- Maximum final context tokens: 2,000.
- Maximum candidates examined: 60.
- Maximum retrieval rounds: 8.

A representation pauses after three consecutive batches add no useful evidence. A large score gap can support pausing only when the batch added no utility. Token, chunk, round, and candidate limits are safety caps, not the target k.

Why this exists: one fixed k cannot suit both a simple fact and a multi-hop question. Multiple signals prevent low-similarity, redundant, or low-value chunks from filling the context.

Better alternative: a learned stopping policy may model sufficiency better, but it changes the research question and requires labeled policy training. The current approach remains deterministic and auditable.

### Step 7 - Deduplicate only after retrieval

After adaptive retrieval stops, merge exact source-unit duplicates, remove contained passages when a retained superset preserves the source, and remove conservative near duplicates only when both source overlap and semantic similarity are high. Preserve all view and rank provenance.

Why this exists: deleting duplicates during probing could remove the agreement signal needed for representation selection. Post-retrieval deduplication keeps the signal but prevents repeated context from reaching the LLM.

Better alternative: maximal marginal relevance can combine selection and diversity online. It is a useful ablation, but it does not preserve the same clean separation between probing, view selection, and final deduplication.

### Step 8 - Make one final Groq call

Send the original question and final evidence with stable E identifiers. The model returns ANSWERABLE, PARTIALLY_ANSWERABLE, or INSUFFICIENT, plus claim-to-evidence mappings. The application renders citations. Invalid or missing mappings cause a safe downgrade to insufficient; there is no repair call.

Why this exists: a language model is useful for composing the final answer, but it is not needed for mathematical retrieval control. One call controls cost and avoids dozens of hidden judgment calls.

Fairness rule: Traditional RAG and SAR-RAG must use the identical answer prompt, JSON schema, Groq model, decoding settings, and output budget. Only retrieval may differ.

## 5. Practical example

Suppose a university policy document says in one section that full-time students may apply, in another paragraph that applicants need six months of enrollment, and in a note that visiting students are excluded. The question is: Who is eligible and under what conditions?

The four top-3 probes return overlapping source units. Paragraph and hierarchical views agree on the eligibility section. The hierarchical view also preserves the heading that limits the rule to a specific program. Paragraph is selected first because it covers the largest RRF mass. Hierarchical is selected second only if its uncovered mass exceeds 0.2437.

Adaptive retrieval accepts the eligibility paragraph, then the six-month condition because it is relevant and novel. It accepts the exclusion note if it adds new information. Repeated windows covering the same sentences are retained during retrieval but merged afterward. Final k might be 3 for this question, even though another question could finish with 1 or reach the safety cap. The LLM receives those three evidence items once and writes a cited answer containing the eligibility rule, condition, and exclusion.

This example explains the algorithm; it is not an experimental result.

## 6. Calibration that was actually completed

Calibration used only official QASPER train and MuSiQue-Full train data. A deterministic bottom-hash sample selected five examples from each of 12 strata defined by dataset, answerability, and document length or hop count. Total sample size was 60.

Each stratum was divided into a 70 percent tuning portion and a 30 percent held-back training-selection portion where possible. Thirty-two SAR configurations and six baseline k values were evaluated without Groq and without gold answer text.

Configurations within 0.01 of the best tuning evidence recall were shortlisted. Held-back selection then maximized complete-evidence recovery, then evidence precision, then minimized context tokens and candidates examined. This avoids an arbitrary weighted score.

The frozen Traditional RAG value is static k = 20. Its 2,000-token budget can produce a final context smaller than 20 chunks; k remains the requested retrieval depth.

On the 24 held-back training-selection examples:

- SAR evidence recall: 0.792.
- Traditional evidence recall: 0.896.
- SAR complete-evidence recovery: 0.688.
- Traditional complete-evidence recovery: 0.875.
- SAR evidence precision: 0.172.
- Traditional evidence precision: 0.113.
- SAR mean context tokens: 1,265.
- Traditional mean context tokens: 1,833.

Interpretation: SAR was more precise and context-efficient, but the baseline recovered more complete evidence. These are calibration diagnostics, not test results. They do not support an overall superiority claim.

## 7. Evaluation design

Both methods use the same question, source documents, MiniLM model, 2,000-token final context budget, Groq answerer, metrics, and execution environment. Traditional RAG uses one sliding-window index and frozen static k = 20. SAR uses the four-view probe and adaptive final k.

Required metrics include exact match, token F1, answerability accuracy/precision/recall/F1, retrieval precision and recall at declared cutoffs, MRR, nDCG, final evidence precision and recall, faithfulness, answer relevancy, per-query latency, offline indexing time, input/output/total tokens, estimated cost, final k, and context tokens.

Faithfulness and answer relevancy from the same Groq model family are secondary because self-evaluation can be biased. Gold-answer and gold-evidence metrics are primary where annotations exist.

Paired bootstrap confidence intervals and paired randomization tests compare methods on the same questions. Statistical significance is not the same as practical importance. Effect sizes, confidence intervals, sample counts, and quality-efficiency trade-offs must all be reported.

QASPER test contains 1,451 normalized questions and is the untouched final QASPER set. QASPER development has already been used for engineering pilots and is not the final claim set. MuSiQue development contains 4,834 locally scorable questions. MuSiQue test labels are hidden, so test predictions require official submission.

The full local evaluation therefore contains 6,285 questions: 1,451 QASPER test plus 4,834 MuSiQue development. With a separate evaluator enabled, this means 18,855 API calls: one Traditional answer call, one SAR answer call, and one evaluator call per question. SAR itself still uses one online LLM call per question.

## 8. What has been reproduced so far

- Full 60-example training-only calibration completed and frozen.
- Frozen manifest contains source hashes, sample IDs, partitions, search space, all trials, and selected controls.
- Eighteen automated tests pass.
- Python dependency check passes in the project virtual environment.
- Real QASPER MiniLM audit: 343 chunks, zero truncations, maximum 231 tokens.
- QASPER test: 1,451 unique normalized IDs.
- MuSiQue development: 4,834 unique normalized IDs, balanced between answerable and insufficient.
- A fair two-question engineering pilot completed with identical answer generators.
- Every pilot question used one Traditional answer call, one SAR answer call, and one evaluator call.

The fair two-question pilot is not evidence of superiority. It is too small and was used to verify the harness. It showed SAR retrieved less context and less total model usage on those cases, but its answer token F1 was lower. No significance claim is valid at n = 2.

## 9. Critical problems found and corrected

- The old pipeline used many candidate-level LLM judgments. It is now a legacy ablation; the primary method uses one final call.
- MuSiQue answerable and insufficient variants had duplicate IDs. Variant suffixes now make IDs unique.
- Raw cosine thresholds were compared across views. They are now representation-specific empirical percentiles.
- Long hierarchical chunks were silently truncated. Chunks are bounded and truncation metadata is recorded.
- Unanswerable examples could inflate retrieval scores. Retrieval metrics are now undefined when no gold evidence denominator exists.
- Search and selection used the same calibration sample. They now use deterministic tuning and held-back training-selection partitions.
- A weighted calibration objective could hide over-retrieval. Selection is now constrained and lexicographic.
- Confirmatory runs could use defaults or a limited prefix. They now require a frozen training manifest and reject baseline-k overrides and limit.
- The baseline and SAR answer prompts differed. Both now use the same answer generator.
- Repeated questions over one document rebuilt indexes. Corpus systems are now reused and offline indexing is measured separately.
- The first reuse cache retained every corpus and could exhaust memory on full MuSiQue. It now retains only the active corpus, which still reuses grouped QASPER questions without benchmark-sized memory growth.
- Long experiments could not safely resume. Runs now save a strict fingerprint and atomic checkpoints; resume is refused when conditions differ.
- Statistical resampling and checkpoints originally scaled poorly. Bootstrap/randomization draws are now processed in bounded-memory batches, and each completed example receives an atomic checkpoint file.
- Groq token prices were hard-coded. Confirmatory runs now require explicitly verified, frozen input, cached-input, and output rates; otherwise cost is not calculated.
- The phrase training-free was inaccurate. The claim is now model-fine-tuning-free and training-calibrated.

## 10. Current scientific conclusion

The implementation now matches the intended pipeline and is ready for a full frozen evaluation. The research idea is reasonable and testable, but the available evidence does not show that SAR-RAG outperforms Traditional RAG overall.

Training diagnostics suggest a trade-off: SAR can remove redundant context and improve evidence precision, while static-k Traditional RAG may recover more complete evidence. The main risk is premature stopping or representation routing that misses a necessary hop. The main potential strength is more focused evidence with fewer context tokens.

The correct claim before full evaluation is: SAR-RAG is an implemented, reproducible adaptive-retrieval method whose quality-efficiency trade-off against Traditional RAG remains unproven.

## 11. Next actions before a paper claim

- Run full QASPER test and full MuSiQue development with the frozen manifest.
- Do not change retrieval thresholds after seeing those results.
- Report both online latency and four-view offline indexing cost.
- Run required ablations: one view, no RRF, no novelty, no gap, no hierarchy, and the legacy multi-call method as a cost ablation.
- Analyze failures by dataset, answerability, hop count, document length, selected views, stop reason, and missing evidence.
- Check overlap and leakage registries before combining dataset results.
- If SAR loses recall, improve the method only in a new development cycle and evaluate the revised method on a new untouched split or clearly label the result exploratory.
- Claim superiority only if the full paired evidence supports a meaningful improvement or a clearly defined Pareto improvement. Otherwise report the honest trade-off.
