# SAR-RAG One-Call Research Pipeline

**Status:** authoritative primary online algorithm  
**Date:** 4 August 2026

## Research claim

SAR-RAG is a model-fine-tuning-free, training-calibrated method that mathematically selects one or two
complementary document representations from a fixed top-3 probe, retrieves adaptively
without a fixed final `k`, removes duplicates after retrieval, and makes exactly one
LLM call for final grounded answering or abstention.

“Training-free” must not be used without qualification: labeled training evidence is
used to calibrate deterministic retrieval controls. No neural model weights are trained
or fine-tuned, and no LLM is used during calibration.

## Fixed primary representations

1. Sentence window.
2. Paragraph.
3. Semantic segment.
4. Hierarchical section.

Paragraph, semantic and hierarchical content groups are bounded to approximately 95
words before embedding. Hierarchical groups repeat their parent heading. The corpus
builder records `original_embedding_tokens` and `embedding_was_truncated` for every
chunk; remaining pathological overlength units are explicitly truncated to MiniLM's
256-token limit rather than silently discarded.

The conventional sliding window is the Traditional RAG baseline and an ablation, not a
fifth primary adaptive view.

## Phase A: ingestion and indexes

Parse every source into stable canonical units. Preserve document ID, version ID,
character offsets, section path and adjacency. Build all four representations from the
same units and embed them with `sentence-transformers/all-MiniLM-L6-v2`.

## Phase B: top-3 mathematical probe

Retrieve ranks 1-3 from every representation. This is diagnostic probe depth, never
final retrieval `k`. Align candidates into evidence groups when their canonical units
are equal, contained, or have Jaccard overlap at least 0.5.

For group `g`:

```text
RRF(g) = sum over representations r retrieving g of 1 / (c + rank_r(g))
NRRF(g) = RRF(g) / (number_of_representations / (c + 1))
```

The primary default is `c=3` because the probe contains three ranks. Freeze `c` on
training data and report `{1, 3, 10, 60}` as a sensitivity analysis.

Select the first representation by maximum total normalized-RRF group coverage. Then
calculate each remaining representation's normalized-RRF mass from groups not covered
by the first. Select a second only when this marginal gain reaches the frozen training
threshold. Resolve ties by lower mean probe-token cost and stable representation ID.

## Phase C: adaptive retrieval without fixed final k

Reuse selected probe candidates. Retrieve subsequent candidates in batches of three.
For candidate `c` and accepted evidence `E`:

```text
relevance(c) = empirical percentile of cosine(query, c) within its representation index
canonical_overlap(c,E) = maximum canonical-unit Jaccard overlap
semantic_similarity(c,E) = maximum embedding cosine to accepted evidence
novelty(c,E) = 1 - canonical_overlap * max(semantic_similarity, 0)
marginal_utility(c,E) = relevance * novelty
```

This percentile normalization is required because raw cosine distributions differ by
representation. Similarity gaps are raw cosine drops divided by that representation
index's score standard deviation. Accept a candidate only when it passes the frozen
relevance-percentile, novelty and marginal-utility
thresholds, is not a high-overlap/high-semantic-similarity duplicate, and fits the
token/chunk safety budgets.

Pause a representation after two consecutive batches add no candidate. A large
similarity gap may pause it earlier only when the same batch adds no marginal utility.
Never stop on similarity gap alone when useful evidence was added.

Global stopping occurs when all selected representations pause or when a candidate,
retrieval-round, final-chunk or tokenizer-token safety cap is reached. These are maximum
limits, not a fixed final retrieval count.

## Phase D: post-retrieval evidence construction

Only after adaptive retrieval stops:

1. Merge exact canonical-unit duplicates and preserve all representation/rank provenance.
2. Remove a contained chunk when a retained superset preserves its source units.
3. Remove near duplicates only when canonical overlap is at least 0.8 and embedding
   similarity reaches the frozen redundancy threshold.
4. Order retained items by marginal utility and compact to the tokenizer-token budget.

The resulting `final_k = len(final_evidence)` is emergent.

## Phase E: exactly one Groq call

Send the original query and final evidence with stable `[E#]` identifiers to
`openai/gpt-oss-120b`. The response must contain answer text, one of `ANSWERABLE`,
`PARTIALLY_ANSWERABLE`, or `INSUFFICIENT`, plus structured claim-to-evidence mappings.
Render inline `[E#]` citations deterministically from those mappings. If the one response
contains missing or invalid mappings, safely downgrade to `INSUFFICIENT`; never make a
repair call or invent a citation.

No LLM is used for query decomposition, probe judgment, representation selection,
adaptive stopping, reformulation or deduplication in the primary method.

## Calibration rule

The implementation defaults are executable starting values, not confirmatory choices.
Freeze RRF constant, representation marginal-gain threshold, similarity threshold,
similarity-gap threshold, novelty threshold, redundancy threshold, marginal-utility
threshold, patience and budgets using training data only. Never change them after
viewing development/test results.

Calibration uses deterministic stratified samples from the official training splits.
Within every stratum, a deterministic held-back training selection partition chooses
among configurations shortlisted on the tuning partition. No Groq call or gold answer
text is used. Retrieval relevance is evaluated against gold evidence spans only.

## Required comparisons

1. Traditional sliding-window dense RAG with training-calibrated static `k`.
2. Primary one-call SAR-RAG.
3. Legacy per-candidate LLM-judged controller as a cost ablation.
4. One-view, no-RRF, no-novelty, no-gap and no-hierarchy ablations.
