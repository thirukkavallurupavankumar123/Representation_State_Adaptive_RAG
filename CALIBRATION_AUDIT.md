# Calibration Audit and Locked Decision Procedure

**Date:** 12 August 2026  
**Scope:** one-call SAR-RAG and Traditional RAG baseline

## Problems corrected before calibration

1. MuSiQue answerable/insufficient pairs shared an `example_id`. The adapter now appends
   `:answerable`, `:insufficient`, or `:hidden`.
2. Raw cosine thresholds were incorrectly assumed comparable across representations.
   Relevance is now an empirical percentile within each representation index; score gaps
   are divided by the index score standard deviation.
3. Long hierarchical chunks were silently truncated by MiniLM. Representations are now
   bounded before embedding and every remaining truncation is recorded.
4. Unanswerable examples were previously capable of receiving perfect retrieval scores
   without gold evidence. Their retrieval precision/recall/ranking metrics are undefined.
5. Hyperparameter search and selection originally used the same sample. Each stratum now
   has deterministic tuning and held-back training-selection partitions.
6. A weighted objective could hide excess retrieval. Selection is constrained and
   lexicographic rather than an arbitrary weighted sum.

## Sampling

- Sources: official QASPER train and MuSiQue-Full train only.
- Seed: `20260803`.
- Streaming bottom-hash sampling, independent of file order.
- Strata: dataset, answerability, and QASPER length or MuSiQue hop count.
- Default: five examples per stratum.
- Each stratum is partitioned 70% tuning / 30% held-back selection where possible.

## Selection rule

1. On tuning data, retain configurations within 0.01 macro gold-evidence recall of the
   best observed configuration.
2. On the held-back training selection data, maximize complete-evidence recovery.
3. Then maximize evidence precision.
4. Then minimize context tokens.
5. Then minimize candidates examined.
6. Use serialized parameter order only as the final deterministic tie-breaker.

This prioritizes evidence coverage but refuses to treat unlimited retrieval as free.

## Interpretation restrictions

- Calibration results are not evaluation results.
- Smoke manifests have `frozen_configuration: null`.
- Gold answers are not used; only evidence annotations and answerability strata are used.
- Validation/test data must not change the frozen configuration.
- Confirmatory evaluation must consume the frozen manifest and the declared evaluation
  JSONL in full; `--limit` and command-line baseline-`k` overrides are rejected.
- The random search does not prove global optimality; full trial results are retained for
  sensitivity and stability analysis.

## Current status

`output/calibration/smoke_manifest.json` is an engineering smoke check, not a frozen
research decision. On its two examples, both methods recovered 0.50 macro gold-evidence
recall; SAR-RAG retrieved more context with lower evidence precision. This tiny result is
not inferential.

The full 60-example stratified training calibration is complete at
`output/calibration/frozen_manifest.json`. It froze SAR-RAG's stopping controls and a
Traditional RAG baseline of `k=20`. On the 24-example held-back training-selection
partition, SAR-RAG used fewer context tokens (1265 versus 1833) and had higher evidence
precision (0.172 versus 0.113), but lower evidence recall (0.792 versus 0.896) and lower
complete-evidence recovery (0.688 versus 0.875). These are calibration diagnostics, not
test results. They do not establish superiority and motivate reporting a quality/efficiency
trade-off unless the untouched evaluation evidence shows otherwise.
