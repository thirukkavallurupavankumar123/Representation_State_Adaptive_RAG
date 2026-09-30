# Reproducible execution commands

These commands are documentation only. The paid command remains blocked until offline retrieval requests are frozen and the user supplies a separately matching authorization file. Scoring remains a later, separately authorized stage because it opens gold answers and evidence labels.

## 1. Re-run local preparation checks

```powershell
& '.venv\Scripts\python.exe' 'tools\dry_run_final_heldout_v1_1.py'
& '.venv\Scripts\python.exe' -m pytest 'tests\test_heldout_v1_1.py' 'tests\test_smoke_generation_v1_1.py' 'tests\test_final_evaluation_protocol.py' -q
```

The dry-run command refuses to overwrite its sealed report. Remove only the disposable dry-run outputs before intentionally reproducing it in a clean copy of the workspace.

## 2. Build final retrieval contexts offline

This command performs deterministic local retrieval only. It makes zero API calls and does not read gold labels.

```powershell
& '.venv\Scripts\python.exe' 'tools\prepare_final_heldout_requests_v1_1.py'
```

It creates `REQUESTS.jsonl`, `RETRIEVAL_AND_PACKING_AUDIT.jsonl`, and `REQUEST_MANIFEST.json`. If any request exceeds the predeclared 2,800 conservative input-token ceiling, it aborts before paid execution; a versioned protocol/cost amendment and new authorization would then be required.

## 3. Paid held-out generation — blocked now

After the request manifest is frozen, replace the disabled proposal with a newly recorded authorization whose exact protocol, run-manifest, and request-manifest fingerprints match. Only then may this command be used:

```powershell
& '.venv\Scripts\python.exe' 'tools\run_final_heldout_v1_1.py' `
  --protocol 'output\final_evaluation\heldout_v1_1\FINAL_EVALUATION_PROTOCOL_AMENDMENT_V1_1.json' `
  --manifest 'output\final_evaluation\heldout_v1_1\FROZEN_QUESTION_SYSTEM_MANIFEST.json' `
  --requests 'output\final_evaluation\heldout_v1_1\REQUESTS.jsonl' `
  --request-manifest 'output\final_evaluation\heldout_v1_1\REQUEST_MANIFEST.json' `
  --checkpoint-dir 'output\final_evaluation\heldout_v1_1\paid_checkpoint' `
  --authorization-file 'output\final_evaluation\heldout_v1_1\PAID_AUTHORIZATION.json' `
  --execute-paid
```

The runner checks the three frozen fingerprints and remaining attempt/cost allowances before each call, uses exact checkpoint fingerprints, and creates no duplicate calls on exact resume.

## 4. Scoring — separately blocked

Gold labels must remain unopened until generation answers and retrieval logs are complete and sealed. The scoring stage must first receive a separate authorization tied to the sealed answer, attempt-ledger, and retrieval-log hashes. No judge model is part of scoring. The planned invocation is:

```powershell
& '.venv\Scripts\python.exe' 'tools\score_final_heldout_v1_1.py' `
  --protocol 'output\final_evaluation\heldout_v1_1\FINAL_EVALUATION_PROTOCOL_AMENDMENT_V1_1.json' `
  --manifest 'output\final_evaluation\heldout_v1_1\FROZEN_QUESTION_SYSTEM_MANIFEST.json' `
  --answers 'output\final_evaluation\heldout_v1_1\paid_checkpoint\ANSWERS.jsonl' `
  --attempts 'output\final_evaluation\heldout_v1_1\paid_checkpoint\API_ATTEMPTS.jsonl' `
  --retrieval-audit 'output\final_evaluation\heldout_v1_1\RETRIEVAL_AND_PACKING_AUDIT.jsonl' `
  --authorization-file 'output\final_evaluation\heldout_v1_1\SCORING_AUTHORIZATION.json' `
  --output 'output\final_evaluation\heldout_v1_1\scoring' `
  --open-gold-and-score
```

The scoring implementation must be regression-tested against the final retrieval-audit schema before this later authorization is requested. It is intentionally not executed during preparation.

