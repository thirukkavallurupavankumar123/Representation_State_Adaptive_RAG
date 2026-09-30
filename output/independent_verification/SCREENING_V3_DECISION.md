# SAR-RAG Screening v3 — Validation, Analysis and Design Decision

**Date:** 14 August 2026
**Scope:** training-only, retrieval-only. Zero LLM calls. No evaluation split touched.
**Basis:** `output/independent_verification/screening_v3_n286/SCREENING_V2_PER_EXAMPLE.csv`,
recomputed from source. No summary JSON was trusted as evidence.
**Analysis code:** `tools/analyze_screening_v3.py` (new; no `src/sar_rag` module was modified).

Every statement below is tagged:
**[RAW]** reproduced measurement · **[STAT]** statistically supported under the frozen
protocol · **[INFER]** reviewer interpretation that goes beyond the test result.

---

## 1. Artifact validation

| Check | Expected | Observed | Verdict |
|---|---|---|---|
| Result rows | 286 × 34 = 9,724 | 9,724 | PASS |
| Unique examples | 286 | 286 | PASS |
| Conditions | — | 34, each with exactly 286 rows | PASS |
| Duplicate `(example, arm, condition)` keys | 0 | 0 | PASS |
| Partition sizes | 194 tuning / 92 selection | 194 / 92 | PASS |
| Stratum sizes | 11×25 + 1×11 (QASPER INSUFFICIENT long) | matches amendment 001 exactly | PASS |
| Calibration-ID overlap | 0 | 0 | PASS |
| Prior-diagnostic-ID overlap (v2, routing, smoke) | 0 | 0 | PASS |
| Split tags | all `train` | 286/286 `train` | PASS |
| Frozen manifest SHA-256 | `5c570ec2b4c52e1b…a54d05` | identical; `git status` clean | PASS |
| LLM/Groq access | none | `RetrievalOnlyAnswerer` only; no API artifact in output | PASS |
| `.env` | untracked | untracked | PASS |

**NaN audit.** 2,584 NaN cells appear in each of `evidence_recall`, `complete_evidence`
and `evidence_precision`. This is **not corruption**: 2,584 = 76 × 34, and the 76 examples
are exactly those with zero gold evidence spans, where recall is undefined by the D01
remediation. Verified cell-by-cell: a metric is NaN *if and only if* `gold_spans == 0`.

**Effective scored sample.** 210 of 286 examples carry gold evidence.

| Partition | QASPER | MuSiQue | Total |
|---|---|---|---|
| **selection (decision basis)** | 22 | 45 | **67** |
| tuning (descriptive) | 48 | 95 | 143 |
| pooled (descriptive) | 70 | 140 | 210 |

> **Subgroup warning.** The QASPER selection cell (n = 22) and the QASPER
> INSUFFICIENT-long stratum (n = 11 total, 4 in selection) cannot support standalone
> claims. Per-dataset selection results are reported but are directional only.

---

## 2. Primary result — selection partition, evidence recall vs frozen full SAR-RAG

n = 67 scored pairs. Holm family = 33 conditions within (metric, partition, scope).

| Condition | mean | Δ vs full | 95% CI | p | Holm p | dz |
|---|---|---|---|---|---|---|
| `sliding_window_k20` | 0.8085 | **+0.0858** | [+0.036, +0.141] | 0.0007 | **0.0165** | +0.39 |
| `sliding_window_matched_k20` | 0.7774 | +0.0547 | [+0.007, +0.109] | 0.0355 | 0.676 | +0.26 |
| `full_hierarchical_v2` | 0.7376 | +0.0149 | [+0.000, +0.045] | 1.000 | 1.000 | +0.12 |
| `candidate_pair_paragraph__hierarchical_v2` | 0.7351 | +0.0124 | [−0.015, +0.050] | 0.754 | 1.000 | +0.09 |
| `semantic_only` | 0.7326 | +0.0100 | [−0.015, +0.045] | 1.000 | 1.000 | +0.08 |
| **`full` (frozen primary)** | **0.7226** | — | — | — | — | — |
| `hierarchical_v2_only` | 0.7189 | −0.0037 | [−0.049, +0.041] | 1.000 | 1.000 | −0.02 |
| `select_one_representation` | 0.7152 | −0.0075 | [−0.022, +0.000] | 1.000 | 1.000 | −0.12 |
| `paragraph_only` | 0.7127 | −0.0100 | [−0.045, +0.015] | 1.000 | 1.000 | −0.08 |
| `no_novelty` / `no_similarity_gap` / `no_post_retrieval_deduplication` | 0.7226 | **0.0000** | [0.000, 0.000] | 1.000 | 1.000 | 0.00 |
| `sliding_window_k10` | 0.6915 | −0.0311 | [−0.070, −0.004] | 0.125 | 1.000 | −0.22 |
| `sentence_window_only` | 0.6629 | −0.0597 | [−0.114, −0.005] | 0.035 | 0.676 | −0.26 |
| `hierarchical_only` (v1) | 0.4726 | −0.2500 | [−0.336, −0.169] | <0.0001 | **0.0016** | −0.72 |
| `rrf_probe_only_no_adaptive` | 0.4092 | −0.3134 | [−0.402, −0.231] | <0.0001 | **0.0016** | −0.87 |

**[STAT]** Exactly one condition beats the frozen primary after Holm correction:
`sliding_window_k20`. **[STAT]** Adaptive retrieval beyond the probe is strongly supported
(`rrf_probe_only_no_adaptive` −0.31, Holm 0.0016) — retrieving past the top-3 probe is the
one component with an unambiguous, large, corrected effect.

Complete-evidence recovery (exact McNemar) agrees: `sliding_window_k20` +0.194,
13 discordant, McNemar p = 0.00024, Holm p = 0.0054.

---

## 3. The budget is not actually shared — and this decides the Traditional comparison

| System | mean context tokens | % of 2,000 cap | mean final_k | recall |
|---|---|---|---|---|
| `full` (SAR-RAG) | 1,305.2 | **65%** | 15.22 | 0.7226 |
| `sliding_window_k20` | 1,857.7 | **93%** | 16.52 | 0.8085 |
| `sliding_window_matched_k20` | 1,780.2 | 89% | 19.30 | 0.7774 |
| `sliding_window_k10` | 1,244.1 | 62% | 9.83 | 0.6915 |

**[RAW]** SAR-RAG stop reasons on selection (n = 92): `all_selected_representations_paused`
63.0%, `final_chunk_safety_cap` 30.4%, `retrieval_round_safety_cap` 6.5%.
**The token budget never appears as a stop reason.** Only 12/92 runs reach ≥95% of the cap.

**[INFER]** The "shared 2,000-token budget" is nominal, not binding. SAR-RAG is halted by
the empty-batch patience rule and the 20-chunk cap while ~700 tokens of its allowance sit
unused; the baseline spends 93% of the same allowance. `sliding_window_k20` therefore wins
**+0.0858 recall while consuming 42.3% more context** (Holm p = 0.0016 on tokens).

**[STAT]** At comparable token spend the advantage disappears in both directions:
`sliding_window_k10` (1,244 tokens) is −0.031 vs SAR-RAG, not significant (p = 0.125);
`sliding_window_matched_k20` is +0.055 but does not survive Holm (p = 0.676).

**Answer — does SAR-RAG have a defensible budget-matched advantage over Traditional RAG?**
**No.** It also is not clearly beaten at matched budget. The matched-budget comparison is
**null**, and the unmatched comparison favours the baseline.

### Stability warning on the `k20` result

| Partition | Δ recall (`sliding_window_k20` − `full`) | p | Holm p |
|---|---|---|---|
| **selection (n=67)** | **+0.0858** | 0.0007 | **0.0165** |
| tuning (n=143, descriptive) | +0.0335 | 0.235 | 1.000 |
| pooled (n=210, descriptive) | +0.0502 | 0.017 | 0.304 |

**[INFER]** The effect is directionally consistent but its magnitude is not stable across
partitions, and it is carried entirely by MuSiQue: **+0.135 (Holm 0.0033) on MuSiQue,
−0.015 (ns) on QASPER**. It should be reported as a serious unresolved threat to the
baseline comparison, not as a finished finding.

---

## 4. Oracle@1 and Oracle@2 — pair complementarity is exactly zero

| Scope | n | Oracle@1 | Oracle@2 | O2 − O1 | best constant view | router | O2 − best constant |
|---|---|---|---|---|---|---|---|
| **selection** | 67 | 0.7575 | 0.7575 | **0.0000** | semantic 0.7326 | 0.7376 | +0.0249 |
| selection · QASPER | 22 | 0.5758 | 0.5758 | 0.0000 | 0.5758 | 0.5758 | +0.0000 |
| selection · MuSiQue | 45 | 0.8463 | 0.8463 | 0.0000 | 0.8111 | 0.8167 | +0.0352 |
| tuning | 143 | 0.8415 | 0.8415 | 0.0000 | 0.8065 | 0.7976 | +0.0350 |
| pooled | 210 | 0.8147 | 0.8147 | 0.0000 | 0.7766 | 0.7785 | +0.0381 |

**[RAW]** Per-example verification across all 210 scored questions and all six forced pairs:

- Candidate system: a pair strictly beats the best single view on **0 of 210** examples.
- Legacy system: **1 of 210** (max gain 0.143), and that example is in the tuning partition.
- A pair is the *unique* maximiser on **0 of 210** examples.

**[STAT]** Oracle@2 exceeds the best constant single view by +0.0249 on selection,
CI [0.000, +0.060], p = 0.251 — **below the frozen 0.05 margin and not significant**.

**[INFER]** This resolves the question that motivated Oracle@2. The earlier Oracle@1 figure
was correctly criticised as not being a ceiling for a one-or-two-view method. Having now
measured the correct ceiling, the correction changes nothing: under a shared budget a second
representation splits the budget and never recovers evidence the best single view misses.
The "select one **or two** complementary representations" premise has **no headroom to
exploit**, not merely a router that fails to exploit it.

**Tie caveat.** 65 of 67 selection examples have ≥2 tied maximisers, so *which* view is
"best" remains unreliable and no per-view attribution is made. The oracle **value** is a
maximum and is unaffected by ties.

---

## 5. Hierarchical v2 — the representation is fixed; the system does not care

**[STAT]** As a standalone view, the repair is decisive and replicates:

| Comparison | Δ recall | 95% CI | p | dz |
|---|---|---|---|---|
| `hierarchical_v2_only` − `hierarchical_only` (selection, n=67) | **+0.2463** | [+0.153, +0.341] | 0.00005 | +0.63 |
| same, tuning (n=143, descriptive) | +0.2476 | [+0.174, +0.321] | 0.00005 | +0.55 |
| same, pooled (n=210, descriptive) | +0.2472 | [+0.189, +0.305] | 0.00005 | +0.58 |

Direct paired tests, not differences of differences. The effect replicates almost exactly
across both partitions.

Hierarchical rises from the worst view (0.4726) to competitive with paragraph (0.7189 vs
0.7127). The mis-specification diagnosis — heading duplication rather than genuine
multi-level retrieval — is confirmed as actionable.

**Frozen retention rule (`hierarchy_candidate_retained`):**

| Requirement | Threshold | Observed | Met |
|---|---|---|---|
| v2 improves over v1 | > 0 | +0.2463 | ✅ |
| candidate primary noninferior to legacy | ≥ −0.02 | +0.0149 | ✅ |
| **and** recall gain | ≥ +0.02 | +0.0149 | ❌ |
| **or** context reduction | ≥ 10% | +0.2% | ❌ |
| **or** Oracle@2 gain over legacy set | ≥ +0.02 | **0.0000** | ❌ |
| **RETAINED** | | | **NO** |

**[INFER]** This is the single most informative result in the experiment. Repairing the
worst representation by **+0.246** recall moves the full system by **+0.015** and moves the
attainable ceiling by **exactly 0.000**. A pipeline whose ceiling is invariant to a
quarter-point improvement in one of its four views is not a pipeline whose value comes from
representation selection.

**Recommendation that departs from the frozen rule's binary output:** `hierarchical_v2` is
a strictly better *implementation* and should replace v1 in the codebase as a correctness
fix, while `hierarchical_v1` is preserved as `hierarchical_v1` for the record. What the rule
correctly denies is the *claim* that the repair rescues the routing contribution.

---

## 6. Gate components — what actually changes an output

Measured across all 286 examples on `final_k`, `context_tokens`, `candidates_examined`,
`retrieval_rounds`, `stop_reason`, recall, precision and complete evidence.

| Component | Examples with ANY output change | Fields changed | Selection recall Δ | Verdict |
|---|---|---|---|---|
| Novelty rule (0.2292) | **0 / 286** | none | 0.0000 | **INERT — remove from claim** |
| Similarity-gap rule (1.9052) | **0 / 286** | none | 0.0000 | **INERT — remove from claim** |
| Post-retrieval deduplication | 56 / 286 | `final_k` 56, `context_tokens` 56, `precision` 26 | 0.0000 | **Efficiency only** |
| Second representation (`select_one_representation`) | 48 / 286 | recall on only **4**, CER on **3** | −0.0075 | Negligible |

**[STAT]** Novelty and similarity-gap removal reproduce the frozen system byte-for-byte on
every measured output at n = 286. This is a far stronger version of the n = 60 v2 finding and
is no longer a power-limited null — it is a structural zero at the frozen thresholds.

### Internal gate instrumentation (n = 286; 6,989 candidates scored, 2,000 batches)

`max_*` fields are aggregated with `max`, not summed — the summation defect present in the
v2 summary artifact is not repeated here.

| Gate | Reached | Would reject | **Binding** |
|---|---|---|---|
| Relevance percentile | 6,989 | 1,788 | **1,788** |
| Novelty (0.2292) | 5,201 | 108 | 107 |
| Marginal utility (0.1289) | — | 135 | **0** |
| Redundancy (0.9547) | — | — | **0** |
| Budget caps | — | — | 478 |

| Similarity-gap rule | Value |
|---|---|
| Eligible batches (empty batch, rule may fire) | 664 |
| Times the rule fired | **0** |
| Times patience preempted it | 0 |
| **Maximum standardized gap ever observed** | **1.0925** |
| Maximum gap on an eligible empty batch | 1.0119 |
| Frozen threshold | **1.9052** |

**[STAT]** The similarity-gap rule is not merely unused — it is **unreachable**. The largest
standardized gap observed anywhere in 286 examples is 1.0925, i.e. 43% below the frozen
threshold of 1.9052. No sample of this data could ever trigger it.

**[STAT]** Marginal utility and redundancy bind **0 times in 6,989 candidate evaluations**.

**[INFER]** Novelty binds 107 times yet changes no final output, which means every candidate
it rejected would have been removed downstream anyway by the budget caps or post-retrieval
deduplication. Novelty is redundant with mechanisms already in the pipeline.

**[STAT]** Deduplication changes `final_k` and `context_tokens` on 56 examples with
**0.0000** change in recall or complete evidence. It is a genuine efficiency component and
must be described as such — never as a quality component.

**[INFER]** Final retrieval behaviour is empirically equivalent to *relevance-percentile-led
adaptive retrieval + patience + safety caps + post-hoc deduplication*. The novelty and
marginal-utility scoring machinery can be deleted at the frozen thresholds with no observable
effect on any output.

---

## 7. Failure analysis (`SCREENING_V3_FAILURES.csv`, 210 scored rows)

| Class | Count |
|---|---|
| Complete evidence recovered | 129 |
| Partial | 55 |
| Total miss (recall = 0) | 26 |

**[RAW]** Of the 26 total misses, **18 are unreachable by any single view or any pair**
(Oracle@2 = 0). Only **14 of 210** examples overall (3 of 67 in selection) would be improved
by perfect routing. Among examples with any routing regret, the stop reason was
`all_selected_representations_paused` in 8 cases and a safety cap in 6.

**[INFER]** The dominant failure mode is that the evidence is not retrievable by *any*
configuration of this embedding + chunking family — an encoder/corpus problem, not a routing
problem. Routing repair addresses at most 6.7% of scored examples.

---

## 8. Frozen decision rules — applied verbatim

### `routing_claim_retained`

| Requirement | Threshold | Observed (selection) | Met |
|---|---|---|---|
| Candidate primary noninferior to best constant view | ≥ −0.02 | +0.0050 | ✅ |
| **and** recall gain | ≥ +0.02 | +0.0050 | ❌ |
| **or** context reduction | ≥ 10% | +0.2% | ❌ |
| **and** Oracle@2 exceeds best constant view | ≥ +0.05 | **+0.0249** (p = 0.251) | ❌ |
| **RETAINED** | | | **NO** |

`pivot_rule` therefore fires: *"stop presenting representation routing as the central
contribution and pivot to the strongest adaptive single-view or budget-matched retrieval
method."*

### Answers to the seven questions posed

| Question | Answer | Basis |
|---|---|---|
| Retain `hierarchical_v2`? | **Yes as a code correctness fix, No as a rescue for the routing claim** | +0.246 as a view; +0.015 system; 0.000 ceiling |
| Is representation routing scientifically justified? | **No** | Fails frozen rule on all three limbs |
| Does Oracle@2 show meaningful pair complementarity? | **No — it is exactly zero** | 0/210 pair wins; O2 = O1 in every scope |
| Does the router approach the oracle ceiling? | **Yes, and that is the problem** | router 0.7376 vs O2 0.7575; regret 0.0199, and the whole ceiling is only +0.025 over a constant policy |
| Is a constant policy stronger? | **Statistically indistinguishable** | semantic 0.7326 vs router 0.7376, Δ +0.005 |
| Do novelty / gap / utility / redundancy / dedup affect outputs? | **novelty, gap: no (0/286). dedup: efficiency only. utility & redundancy: never bind** | §6 |
| Defensible budget-matched advantage over Traditional RAG? | **No** | §3; matched-budget null, unmatched loss |

---

## 9. FINAL DECISION

> ## **PIVOT TO ADAPTIVE SINGLE-VIEW RETRIEVAL**

Representation-state routing is not defensible as the paper's central contribution. It is
not that the router is badly tuned — the attainable ceiling over all four views and all six
pairs sits **+0.025** above simply always using the semantic view, the frozen protocol
required **+0.05**, and no pair ever beat the best single view on any of 210 questions.
Repairing the weakest view by +0.246 moved that ceiling by 0.000.

This is a **decisive negative result, not an inconclusive one.** The Oracle@2 finding is a
structural zero measured at n = 210, not a power-limited null.

### What survives and should become the v2 method

1. **Adaptive retrieval past a shallow probe** — the one large, Holm-corrected, unambiguous
   effect (`rrf_probe_only_no_adaptive` −0.313, Holm 0.0016).
2. **A single representation** — `semantic` on current evidence, with `paragraph` and the
   repaired `hierarchical_v2` statistically indistinguishable from it.
3. **Post-retrieval deduplication** as an explicit efficiency mechanism.
4. **Relevance-percentile scoring**, which is the only gate that ever binds.

### What must be dropped from the claim

RRF representation selection; the one-or-two-view premise; the novelty rule; the
standardized similarity-gap rule; marginal utility and redundancy as contributing
components; and any assertion of superiority over Traditional RAG.

### Blocking prerequisites before any paid run

1. **The budget defect must be fixed first.** SAR-RAG spends 65% of its stated token budget
   while the baseline spends 93%. Every Traditional comparison in this project is confounded
   until the two arms are matched on *tokens actually consumed*, not on chunk count. This
   must be re-screened, free, before any LLM spend.
2. **Recalibration is mandatory, not optional.** The frozen thresholds were fitted with the
   four-view router and hierarchical v1 in the pipeline. A single-view method is a different
   estimand; the v1 manifest cannot carry over.
3. The `sliding_window_k20` instability (selection +0.086 / tuning +0.034) must be resolved
   on training data.

**The paid held-out evaluation remains blocked.** Nothing in this analysis authorises it,
and running it against the current method would spend the budget confirming a contribution
this experiment has just refuted.

---

## 10. Deliverables

| File | Contents |
|---|---|
| `SCREENING_V3_VALIDATION.json` | artifact validation, oracles by scope, gate instrumentation, frozen rules applied |
| `SCREENING_V3_RESULTS.csv` | 1,485 comparisons: Δ, bootstrap CI, randomization p, Holm p (all-conditions and four-single-view families), dz, paired n, exact McNemar |
| `SCREENING_V3_FAILURES.csv` | 210 scored examples with per-example oracle, regret, tie multiplicity, stop reason and recoverability |
| `SCREENING_V3_DECISION.md` | this document |
| `SAR_RAG_RESEARCH_STATUS.pdf` | updated research-status report |

**Preserved unmodified:** `output/calibration/frozen_manifest.json` (SHA-256 verified),
all version-1 artifacts, `screening_v3_n286/` raw outputs.
