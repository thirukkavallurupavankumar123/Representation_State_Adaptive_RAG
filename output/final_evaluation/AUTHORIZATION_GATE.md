# Authorization gate

No paid operation is authorized by this file. Before the first Groq request, all of the following must hold:

1. The user explicitly authorizes the 48-call paid smoke test.
2. `PAID_SMOKE_AUTHORIZATION.json` is created after that authorization and matches protocol fingerprint `411bf416e795ac63559ce1a387ab77c8c8494c579a35c65125ab9081dbf0703e` and smoke manifest fingerprint `43b3b9c787013e03cf284aba6cc584c4f8f8cf9b023e2621e5881055233725c2`.
3. Amendment 001 fixes the cumulative ceiling at exactly 48 API attempts including retries and disables automatic retries because all 48 attempts are needed for the 48 frozen pairs.
4. Its maximum estimated charge is no more than $0.04320 under the frozen rates.
5. The prepared request file passes fingerprint and schema validation.
6. The command includes the deliberate `--execute-paid` flag.

The full held-out run and optional LLM judge each require a later, separate authorization. Smoke authorization never authorizes either stage.

After explicit user authorization, the authorization file must contain exactly:

```json
{
  "status": "EXPLICIT_USER_AUTHORIZATION_RECORDED",
  "authorized_stage": "paid_smoke_only",
  "user_authorized": true,
  "protocol_fingerprint": "411bf416e795ac63559ce1a387ab77c8c8494c579a35c65125ab9081dbf0703e",
  "smoke_manifest_fingerprint": "43b3b9c787013e03cf284aba6cc584c4f8f8cf9b023e2621e5881055233725c2",
  "request_manifest_fingerprint": "c6aba531e02404a00f903724e7ca4fa2532420c4c1df81ab5aaf0c82c72eda42",
  "authorization_amendment_sha256": "736bddbe1a22cce779cf557b530288a265fcc5de780d71bd9fa9d1cf91e4e7b7",
  "maximum_total_api_attempts_including_retries": 48,
  "maximum_retries_per_request": 0,
  "maximum_cost_usd": 0.0432
}
```

The exact command that would begin the paid smoke test is:

```powershell
& '.venv\Scripts\python.exe' 'tools/run_final_answer_evaluation.py' --mode paid-smoke --protocol 'output/final_evaluation/FINAL_EVALUATION_PROTOCOL.json' --manifest 'output/final_evaluation/SMOKE_TEST_MANIFEST.json' --requests 'output/final_evaluation/SMOKE_TEST_REQUESTS.jsonl' --request-manifest 'output/final_evaluation/SMOKE_TEST_REQUESTS_MANIFEST.json' --checkpoint-dir 'output/final_evaluation/paid_smoke_checkpoint' --authorization-file 'output/final_evaluation/PAID_SMOKE_AUTHORIZATION.json' --execute-paid
```

This command has **not** been executed.
