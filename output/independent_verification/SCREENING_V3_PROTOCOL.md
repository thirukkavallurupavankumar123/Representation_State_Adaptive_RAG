# Frozen Training-Only Screening V3 Protocol

Status: **frozen before execution**. This is a retrieval-design experiment, not held-out test evidence and not an LLM answer-quality experiment.

The run uses 300 fresh official training examples: 25 from each of 12 QASPER/MuSiQue strata. All calibration, previous screening and smoke-test IDs are excluded explicitly. Within every stratum, 17 examples form the tuning partition and 8 form the untouched selection partition. Design decisions use the selection partition; pooled results are descriptive.

No evaluation split may be opened. No Groq client or other LLM is allowed. Embeddings remain the locally verified 384-dimensional `all-MiniLM-L6-v2` model.

## Compared systems

The legacy primary system is reproduced unchanged. The candidate primary differs only by replacing legacy `hierarchical` with separately versioned `hierarchical_v2`. The legacy implementation remains available as an ablation.

The hierarchy candidate fuses child and parent-section percentile scores with predeclared equal weights. This is a neutral candidate, not an optimality claim. Both four-view systems are accompanied by all four single views and all six forced representation pairs under identical shared budgets. Traditional sliding-window controls use k = 3, 5, 10 and 20 at original and matched granularity.

Oracle@1 is the label-only maximum over single views. Oracle@2 is the label-only maximum over the same singles and all forced pairs. Neither oracle is deployable or a generalization estimate. Every tied maximizer is retained.

## Outcomes and inference

Primary retrieval outcomes are unique gold-span recall, complete-evidence recovery and final context tokens. Evidence precision, final k, candidates, rounds, stop reasons, index size and indexing time are secondary.

Every comparison is paired. Report bootstrap 95% confidence intervals, paired-randomization p-values, paired effect size and valid n. Complete-evidence recovery additionally uses exact McNemar. Holm correction is applied by metric and scope across the complete condition family; four-view single-representation sensitivity families are also shown transparently.

Practical margins are frozen at -0.02 for recall/CER noninferiority, +0.02 for a meaningful recall improvement, 10% for meaningful context reduction and +0.05 for a materially useful Oracle@2 ceiling over the best constant single view.

## Decision rule

Retain the hierarchy candidate only if it improves over hierarchy v1 and the candidate primary is recall-noninferior to legacy full SAR-RAG on the selection partition. It must additionally provide a meaningful primary recall gain, a meaningful context reduction under noninferiority, or at least +0.02 Oracle@2 gain.

Retain representation routing as the central contribution only if the candidate router is recall-noninferior to the best constant view and either improves recall by at least 0.02 or reduces context by at least 10%. Oracle@2 must exceed the best constant view by at least 0.05. Otherwise the research pivots to the strongest adaptive single-view or budget-matched retriever.

A gate is not a contributing mechanism merely because it is evaluated or rejects a candidate. Removing it must change final output without violating the -0.02 recall/CER margins.

The complete machine-readable specification and exact command are in `SCREENING_V3_PROTOCOL.json`.
