# Sealed held-out v1.1 commands

All paths are relative to the workspace root. The generation command remains blocked unless both `--execute-paid` and a separately authorized, fingerprint-exact authorization file are supplied.

## Revalidate offline package

```powershell
& '.\.venv\Scripts\python.exe' tools/seal_heldout_requests_v1_1.py
& '.\.venv\Scripts\python.exe' tools/validate_sealed_heldout_execution_v1_1.py
& '.\.venv\Scripts\python.exe' -m pytest tests/test_heldout_v1_1.py -q
```

The seal command is intentionally create-once and will refuse to overwrite sealed artifacts.

## Paid generation (not authorized by this document)

```powershell
& '.\.venv\Scripts\python.exe' tools/run_final_heldout_v1_1.py `
  --protocol output/final_evaluation/heldout_v1_1/FINAL_EVALUATION_PROTOCOL_AMENDMENT_V1_1.json `
  --manifest output/final_evaluation/heldout_v1_1/FROZEN_QUESTION_SYSTEM_MANIFEST.json `
  --requests output/final_evaluation/heldout_v1_1/REQUESTS.jsonl `
  --request-manifest output/final_evaluation/heldout_v1_1/REQUEST_MANIFEST_SEALED.json `
  --authorization-file output/final_evaluation/heldout_v1_1/AUTHORIZATION.json `
  --checkpoint-dir output/final_evaluation/heldout_v1_1/generation_checkpoint `
  --execute-paid
```

## Scoring (separately gated; do not run during generation authorization)

```powershell
& '.\.venv\Scripts\python.exe' tools/score_final_heldout_v1_1.py --help
```
