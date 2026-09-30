# Final answer-generation protocol handoff

Status: **verified and frozen; waiting for explicit paid-smoke authorization**.

- Phase E values were recomputed from 63 paired per-example rows; v2 remains unsupported as a v1 replacement.
- Smoke sample: 16 design-exposed training questions, balanced 8/8 by dataset and answerability.
- Systems: Traditional sentence-window k=10, unchanged frozen SAR-RAG v1, frozen paragraph v2.
- Prepared requests: 48; fresh retrieval recomputation was byte-substantively identical.
- Deterministic dry run: 48 fake calls, 48 checkpoint records, zero new calls on resume, fingerprint drift rejected, zero failures/retries, zero Groq calls.
- Complete suite: 113 passed, 0 failed (14 third-party deprecation warnings).
- Paid smoke maximum: 48 calls; expected 112,800 tokens and $0.02232; planning ceiling 172,800 tokens and $0.04320.
- Full held-out and optional judge stages remain separately blocked.

See `AUTHORIZATION_GATE.md` for the exact paid command. It was not executed.
