# Frozen final answer-generation evaluation protocol

Status: **FROZEN BEFORE ANY PAID EXECUTION**  
Protocol fingerprint: `411bf416e795ac63559ce1a387ab77c8c8494c579a35c65125ab9081dbf0703e`

The study compares Traditional sentence-window RAG at frozen k=10, unchanged frozen SAR-RAG v1, and frozen paragraph-based Adaptive Single-View v2. All use `openai/gpt-oss-120b`, the same strict evidence-only prompt/schema, temperature 0, top-p 1, medium reasoning, an 800-token output cap, and a common 1,600-token packed evidence cap. Traditional k=20 remains a declared retrieval-only Pareto comparator and receives no answer call.

The paid smoke test contains 16 already design-exposed training questions: eight QASPER and eight MuSiQue; eight answerable and eight insufficient. It requires exactly 48 calls because cross-system response sharing is prohibited. It is operational validation only.

Final held-out scoring uses QASPER official test (1451 ID-only records) and MuSiQue-Full validation (4834 ID-only records), for 6285 questions and at most 18855 answer calls. Predictions must be sealed before labels are opened. No final outcome may change retrieval, prompt, decoding, or analysis rules.

Primary outcomes are answer token F1, unique gold-evidence recall, answerable-only complete-evidence recovery, and actual context tokens. Paired bootstrap intervals, paired randomization tests, exact McNemar tests, and Holm correction are frozen in the JSON protocol, along with practical margins and interpretation rules.

No paid call and no held-out label access occurred while creating this protocol.
