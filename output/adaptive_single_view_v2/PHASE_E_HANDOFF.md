# SAR-RAG corrected screening and v2 handoff

## Decision

- RRF one-or-two-view routing is rejected for the tested candidate family: candidate pair gain is zero across all 210 scored development examples.
- The frozen v2 selection is paragraph retrieval with a 1,600-token operating point, relevance percentile 0.45, patience 1, six-round cap, 30-candidate cap, and 20-chunk cap.
- V2 improves recall over its shallow probe and same-view fixed-k=10 comparator, while using more context than fixed-k=10.
- V2 uses 17.6% fewer tokens than same-view fixed-k=20 under the preregistered recall-noninferiority comparison.
- The overall v2 replacement category is **UNSUPPORTED** because frozen v1 point-dominates v2 on the fresh development-selection sample; neither paired v2-minus-v1 difference is significant.
- Held-out answer quality remains unknown. No paid, Groq, or other LLM calls were made.

## Primary artifacts

- `output/independent_verification/SCREENING_V3_RESULTS_CORRECTED.csv`
- `output/independent_verification/SCREENING_V3_VALIDATION_CORRECTED.json`
- `output/independent_verification/SCREENING_V3_CORRECTION_NOTE.md`
- `output/adaptive_single_view_v2/PROTOCOL.json`
- `output/adaptive_single_view_v2/PROTOCOL_AMENDMENT_001.json`
- `output/adaptive_single_view_v2/recalibration_seed20260905/FROZEN_V2_MANIFEST.json`
- `output/adaptive_single_view_v2/recalibration_seed20260905/PARETO_AND_DECISION.json`
- `output/adaptive_single_view_v2/recalibration_seed20260905/RESEARCH_DECISION.md`
- `output/pdf/SAR_RAG_RESEARCH_STATUS_V2.pdf`

See `REPRODUCTION_COMMANDS.md` for the exact commands.

## Validation

- Complete suite: **104 passed, 0 failed** in 47.22 seconds (14 third-party deprecation warnings).
- Frozen v1 manifest SHA-256: `5c570ec2b4c52e1b0ea0f2152a6f8fb6f1fa1c1575e58cfb58bde2c434a54d05` (matches expected).
- Protected tracked differences in the frozen v1 manifest and raw screening directory: none.
- `.env` remains untracked; its contents were not read.
- Recalibration manifest: zero LLM calls, zero failures, zero retries.
- Updated PDF: seven pages, all rendered and visually inspected.
- Full worktree inventory: `WORKTREE_FILE_INVENTORY.txt` (151 entries at validation time).

## Blocked work

Paid smoke testing and final held-out answer-generation evaluation remain blocked pending explicit authorization, frozen prompt/model fingerprints, a call/cost ceiling, and a separately frozen held-out evaluation plan.
