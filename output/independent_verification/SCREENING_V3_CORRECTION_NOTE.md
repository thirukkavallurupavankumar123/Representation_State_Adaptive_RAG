# Screening v3 correction note

**Scope:** versioned correction computed directly from the preserved 9,724-row per-example CSV.
The historical screening outputs were not overwritten. No LLM or held-out split was used.

## Corrected metric populations

- Unique gold-span recall remains defined over all 210 questions with one or more gold spans.
- Complete-evidence recovery (CER) is now restricted to answerable questions. The selection population is 46.
- For insufficient-evidence questions that retain residual support annotations, complete recovery of those residual spans is reported separately. The selection population is 21.
- Questions with no gold spans remain unscored for evidence recall/precision/CER; the 2,584 NaNs per metric are expected.

## Oracle complementarity correction

- Hierarchical-v2 candidate family: Oracle@2 - Oracle@1 is exactly 0.000000 across 210 scored examples; 0 examples have a positive pair gain.
- Legacy family: selection gain is 0.000000; pooled gain is 0.000680, caused by 1 tuning example with maximum gain 0.142857.

Therefore, “pair complementarity is exactly zero in every scope” was too broad.

## Actual-token comparison correction

The 2,000-token value is a safety maximum, not a consumption target. Fair interpretation uses the same queries, reports actual consumed tokens, shows the recall-token Pareto frontier, and declares a matched operating point as mean consumption within 10% of SAR-RAG. Matched traditional points: traditional:sliding_window_k10.

`sar_has_significant_matched_actual_token_advantage` is **false**. This explicitly tests a positive SAR advantage. The previous `budget_matched_defensible` name was ambiguous because “no tested traditional condition significantly beats SAR” does not establish that SAR significantly beats a matched-token comparator.

## Repository and frozen artifacts

The working tree is not clean. The defensible integrity statement is narrower: the frozen v1 manifest SHA-256 is `5c570ec2b4c52e1b0ea0f2152a6f8fb6f1fa1c1575e58cfb58bde2c434a54d05` and it matches the preregistered expected hash. No claim of a clean Git worktree is made.

## Scoped interpretation

The development screening rejects the tested RRF one-or-two-view routing claim only for the current encoder, representation family, datasets, retrieval configuration, and design-exposed sample. Unreachable failures are evidence about the tested retrieval system, not proof that the encoder or routing can never work elsewhere. These results are not held-out evidence.
