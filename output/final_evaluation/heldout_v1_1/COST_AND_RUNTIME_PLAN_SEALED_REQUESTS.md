# Sealed-request calls, cost, and runtime

Pricing remains the frozen official `openai/gpt-oss-120b` rate: $0.15/M uncached input tokens and $0.60/M output tokens. All figures below are estimates, not actual invoiced charges.

| Quantity | Sealed value |
|---|---:|
| Requests / initial attempts | 18,720 |
| Maximum attempts (3 per pair) | 56,160 |
| Actual preflight input tokens, initial | 27,459,247 |
| Conservative billed input reserve, initial | 29,855,407 |
| Reserved output tokens, initial | 29,952,000 |
| Reserved output tokens, maximum | 89,856,000 |
| Smoke-mean planning estimate, initial | $9.77419905 |
| Conservative initial reserve | $22.44951105 |
| Final conservative authorization ceiling | $67.34853315 |

No sealed request exceeds the 2,800-token conservative input limit or the 1,600-token supplied-context/completion limits. The runner checks the remaining attempt and exact per-request reserved-cost allowance before every call.

The frozen serial pacing of 30 seconds between attempt starts gives a no-stall lower bound of 156.00 hours (6.50 days) for the initial round and 468.00 hours (19.50 days) at the retry ceiling. Account RPM/TPM/RPD/TPD limits remain unknown and may increase elapsed time; they cannot reduce this serial pacing lower bound.
