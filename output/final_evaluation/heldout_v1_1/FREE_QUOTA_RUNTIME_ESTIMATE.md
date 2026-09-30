# Free-quota runtime estimate

Published free limits for `openai/gpt-oss-120b`: 30 RPM, 1,000 RPD, 8,000 TPM, and 200,000 TPD. Groq says limits are organization-level and exceptions may exist; the private console override could not be checked because the available browser was signed out.

| Scenario | Tokens | TPD-dominated quota windows |
|---|---:|---:|
| Initial, smoke-mean output | 36,884,767 | 185 days |
| Initial, conservative 1,600 output | 59,807,407 | 300 days |
| Three-attempt maximum, smoke-mean output | 110,654,301 | 554 days |
| Three-attempt maximum, conservative | 179,422,221 | 898 days |

TPD dominates RPD (19 initial request-days and 57 maximum request-days). Calendar completion cannot be shorter than the quota-window count and may be longer because quota is organization-wide, other usage consumes the same limits, actual reset boundaries are provider-controlled, and persistent failures can require retries. The runner uses conservative rolling 24-hour accounting when a provider daily reset is not exposed.

Generation remains disabled. Billing, plan upgrades, and credit purchases are prohibited; maximum authorized billed cost is $0.
