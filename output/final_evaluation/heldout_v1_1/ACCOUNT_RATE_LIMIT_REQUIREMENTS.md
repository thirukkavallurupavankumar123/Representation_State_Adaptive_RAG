# Account-specific Groq limits still required

The Groq console limits page was checked read-only without a generation call, but the available browser session was signed out. No account-specific limit could be verified.

Before paid authorization, provide the effective limits for the exact organization/project and model `openai/gpt-oss-120b`:

- requests per minute (RPM)
- requests per day (RPD)
- tokens per minute (TPM), including whether input/output have separate limits
- tokens per day (TPD), including whether input/output have separate limits
- the applicable plan/tier and any project-level overrides
- whether failed or rate-limited attempts count against each limit

The executable remains serial with a frozen 30-second minimum interval. Limits are needed to estimate quota stalls and confirm that the initial 18,720 requests and maximum 56,160 attempts can complete within the intended calendar window. No generation call will be used to discover these values.
