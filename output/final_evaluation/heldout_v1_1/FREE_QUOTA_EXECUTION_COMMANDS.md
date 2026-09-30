# Free-quota held-out execution commands

The command remains blocked until `AUTHORIZATION_FREE_QUOTA.json` exactly matches the frozen free-quota authorization schema. Each invocation runs until completion or a quota pause. After a pause, read `generation_checkpoint/FREE_QUOTA_STATE.json` and rerun only after `resume_not_before_utc`.

```powershell
& '.\.venv\Scripts\python.exe' tools/run_final_heldout_v1_1.py `
  --protocol output/final_evaluation/heldout_v1_1/FINAL_EVALUATION_PROTOCOL_AMENDMENT_V1_1.json `
  --manifest output/final_evaluation/heldout_v1_1/FROZEN_QUESTION_SYSTEM_MANIFEST.json `
  --requests output/final_evaluation/heldout_v1_1/REQUESTS.jsonl `
  --request-manifest output/final_evaluation/heldout_v1_1/REQUEST_MANIFEST_SEALED.json `
  --free-quota-plan output/final_evaluation/heldout_v1_1/FREE_QUOTA_EXECUTION_PLAN.json `
  --authorization-file output/final_evaluation/heldout_v1_1/AUTHORIZATION_FREE_QUOTA.json `
  --checkpoint-dir output/final_evaluation/heldout_v1_1/generation_checkpoint `
  --execute-paid
```

`--execute-paid` is the historical network-execution safety flag; the bound authorization requires free quota only, $0 maximum billed cost, and forbids billing, upgrades, or credit purchases.
