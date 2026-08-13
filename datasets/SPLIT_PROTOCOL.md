# Dataset split protocol

The experiment must freeze all online controls before looking at evaluation labels.

| Dataset | Calibration | Local validation/evaluation | Final use |
|---|---|---|---|
| QASPER v0.3 | Train only | Development | Test |
| MuSiQue-Full v1.0 | Train only | Development | Official test submission only; labels are hidden |
| Custom representation-state benchmark | Authored calibration partition | Audited held-out partition | Held-out partition |
| HotpotQA | None | None | External generalization only, if added later |

The deterministic `--limit N` option takes the first N records and is only for pipeline
smoke testing. It must not be reported as a representative experimental subset. A real
calibration sample must be selected from train with a recorded seed and stratified by
answerability, hop count/question type, document length, and evidence count where those
labels exist.

The following are frozen after calibration:

- representation definitions and chunk controls;
- probe depth (`p=3`; `p=5` is an ablation only and cannot replace the primary setting);
- RRF constant selected from `{1, 3, 10, 60}`; `k0=3` is the registered anchor
  because the diagnostic list has only three ranks;
- adaptive batch size (`3`);
- representation marginal-gain, similarity, similarity-gap, novelty, redundancy and
  marginal-utility thresholds;
- no-gain patience plus candidate, retrieval-round, final-chunk and tokenizer-token budgets;
- Groq final-answer model/prompt/schema and embedding revision.

Calibration is retrieval-only and makes no LLM calls. Select configurations
lexicographically: retain configurations within 0.01 of the best macro gold-evidence
recall; maximize complete-evidence rate, then evidence precision; finally minimize
context tokens and candidates examined. This prevents an arbitrary weighted score from
hiding retrieval/efficiency tradeoffs.

No final retrieval `k` is calibrated. `final_k` remains an output of evidence sufficiency,
reactivation, post-retrieval deduplication, and context compaction.
