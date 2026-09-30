# Screening V3 Protocol Amendment 001

This amendment was recorded after deterministic sample counting and before any embedding, retrieval, metric or inference result was produced.

The frozen protocol requested 25 examples from each of 12 training strata. After excluding all 189 examples previously used in calibration or diagnostics, the QASPER insufficient/long stratum contains only 11 eligible examples. Every other stratum has at least 25. The feasible sample is therefore 286, not 300.

The sampling rule remains 25 per stratum, exhausting a stratum when fewer than 25 eligible examples exist. The 11 undersized examples partition deterministically into 7 tuning and 4 selection cases. Every 25-example stratum retains the frozen 17 tuning and 8 selection allocation. Expected totals are 194 tuning and 92 selection.

No result-dependent choice occurred. All models, conditions, outcomes, tests, margins and decision rules remain unchanged. Results must disclose the unequal stratum size, and the 11-example subgroup cannot support a strong standalone conclusion.

The amended run is written to `output/independent_verification/screening_v3_n286`; the original frozen protocol remains preserved unchanged.
