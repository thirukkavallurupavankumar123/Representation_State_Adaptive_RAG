# Adaptive Single-View Retrieval V2 - preregistered protocol

**Status:** FROZEN_BEFORE_IMPLEMENTATION  
**Version:** adaptive-single-view-v2.0.0  
**Frozen:** 2026-09-05  
**Scope:** training-only retrieval; no LLM calls and no held-out access.

## Research question and hypotheses

The study asks whether a deterministically configured adaptive single-view retriever can preserve or improve evidence retrieval while using context more efficiently than shallow-probe and fixed-k comparators. Meaningful effects are preregistered as +0.02 recall, +0.03 answerable-only CER, or at least 10% fewer actual context tokens under a -0.02 recall noninferiority margin. Negative and null outcomes must be reported.

## Data partitions and exposure control

All 7766 previously inspectable IDs are frozen in `DESIGN_EXPOSED_IDS.json` and excluded. A fresh 225-example training-only sample uses nine evidence-bearing strata: three QASPER answerable length strata and six MuSiQue answerability/hop strata. Deterministic within-stratum hashing assigns 162 tuning and 63 selection examples before outcomes are computed. QASPER insufficient examples are excluded because they have no gold evidence spans; MuSiQue insufficient examples retain residual support and are analyzed separately.

QASPER test and all other final held-out records are forbidden. V2 IDs must contain `:train:`; final QASPER IDs contain `:test:`, so disjointness can be verified structurally without opening held-out records.

## Method

Each run uses exactly one declared representation: semantic, paragraph, hierarchical_v2, or sentence-window. There is no per-query router, RRF, or representation pair. Retrieval starts with a depth-3 probe, continues in batches of three, uses deterministic percentile-normalized relevance, and stops via calibrated no-gain patience plus token, round, candidate, and chunk safety caps. Novelty, similarity-gap, marginal-utility, and redundancy gates are absent from the v2 primary path. Deduplication occurs after retrieval. Final k remains emergent.

Token operating points are 800, 1,200, and 1,600 tokens, with 2,000 as a safety maximum only. All embeddings use the frozen local all-MiniLM-L6-v2 snapshot. No LLM may be called in retrieval, parameter fitting, or selection.

## Outcomes and comparisons

Primary retrieval outcomes are unique gold-span recall, answerable-only CER, and actual context tokens. Answer token F1 is deferred until paid authorization. Secondary outcomes include precision, insufficient-question residual-support recovery, MRR, nDCG with a chunk-granularity caveat, k, latency, and stop reasons.

Comparisons use identical queries, embeddings, and maximum context allowance. Reports must show actual consumption, recall-token Pareto frontiers, and matched actual-token points (mean tokens within 10%). Adaptive retrieval is not required to fill the safety budget.

## Statistics and selection

Paired bootstrap and paired randomization use 20,000 resamples with seed 20260905; binary outcomes use exact McNemar tests; declared primary comparisons receive Holm correction. Parameters are fit on tuning only. On selection, only the twelve tuning-selected representation/token candidates are eligible. Candidates within 0.01 of best recall are tied, then answerable CER is maximized, actual tokens and candidates are minimized, and the preregistered representation/token order breaks exact ties. Prior semantic screening performance provides no bonus.

The final interpretation must be one of: better on retrieval quality, better only on efficiency, Pareto competitive, or unsupported, using the exact rules in `PROTOCOL.json`.

## Integrity, resume, and paid gates

Every run records commands, configuration, seed, versions, hashes, failures, retries, and fingerprints. Resume is rejected on any protocol, configuration, embedding, input, sample, seed, or code mismatch. Outputs from distinct configurations are never mixed.

A paid smoke test is allowed only after retrieval artifacts, tests, exact frozen-manifest reproduction, zero-overlap audit, frozen answer prompt/model, and explicit user authorization with call and cost ceilings. It may use design-exposed training examples only. Final held-out evaluation requires a successful smoke, unchanged hashes, an exact cost/resume plan, and separate explicit authorization.
