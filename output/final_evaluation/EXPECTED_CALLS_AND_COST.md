# Expected calls and cost

Pricing frozen on 2026-09-05 from Groq's official model page for `openai/gpt-oss-120b`: $0.15/M input tokens, $0.075/M cached input tokens, and $0.60/M output tokens. Estimates conservatively assume no cache discount.

| Stage | Questions | Systems | Calls | Expected tokens/call (in/out) | Hard planning ceiling (in/out) | Expected cost | Ceiling cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Paid smoke | 16 | 3 | 48 | 2,100 / 250 | 2,800 / 800 | $0.02232 | $0.04320 |
| Final held-out answers | 6285 | 3 | 18855 | 2,100 / 250 | 2,800 / 800 | $8.77 | $16.97 |

Token totals:

- Paid smoke expected: 100,800 input + 12,000 output = **112,800 tokens**. Planning ceiling: 134,400 input + 38,400 output = **172,800 tokens**.
- Final held-out expected: 39,595,500 input + 4,713,750 output = **44,309,250 tokens**. Planning ceiling: 52,794,000 input + 15,084,000 output = **67,878,000 tokens**.

The 2,800-input figure is an authorization/cost planning ceiling, not an instruction to pad prompts. Prepared evidence is rejected before submission if it exceeds the exact 1,600-token evidence cap; it is never silently truncated. Actual provider input usage is recorded after each response. If it exceeds 2,800 tokens, the smoke test fails and no later stage is permitted until a versioned protocol amendment and new authorization.

The optional blinded comparative LLM-judge stage is separate, has at most 6285 additional calls, and remains unauthorized. Its cost must be re-estimated at the time of a separate authorization.
