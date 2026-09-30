# Calls, cost, and runtime

Official Groq pricing verified on 2026-09-26 for `openai/gpt-oss-120b` is $0.15 per million input tokens, $0.075 per million cached input tokens, and $0.60 per million output tokens. Estimates assume no cache discount.

| Quantity | Frozen value |
|---|---:|
| Eligible questions | 6,240 |
| Systems | 3 |
| Initial API attempts | 18,720 |
| Maximum attempts per pair | 3 |
| Maximum API attempts | 56,160 |
| Expected input tokens, initial round | 29,599,830 |
| Expected output tokens, initial round | 9,425,520 |
| Expected total tokens, initial round | 39,025,350 |
| Estimated initial cost | $10.09528650 |
| Conservative initial-round reserve | $25.83360000 |
| Conservative maximum authorization ceiling | $77.50080000 |

The expected figures extrapolate the successful 48-pair v1.1 smoke means. The conservative reserve assumes 2,800 input tokens and the full 1,600 completion tokens for every attempt. These are planning estimates, not actual invoiced charges; actual charges must be verified in Groq billing after execution.

At the frozen 30-second minimum spacing, a serial initial round takes at least 156 hours (6.5 days), and the three-attempt maximum takes 468 hours (19.5 days), before any daily quota stalls. Groq's public limits page shows account/tier-dependent limits; under a 1,000-request/day limit the corresponding lower bounds are 18.72 and 56.16 days. Exact organization limits remain an operational prerequisite to authorization.

