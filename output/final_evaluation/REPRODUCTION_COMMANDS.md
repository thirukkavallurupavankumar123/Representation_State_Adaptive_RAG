# Reproduction commands

```powershell
& '.venv\Scripts\python.exe' 'tools/freeze_final_evaluation_protocol.py'
& '.venv\Scripts\python.exe' 'tools/prepare_smoke_test_requests.py'
& '.venv\Scripts\python.exe' 'tools/prepare_smoke_test_requests.py' --verify-existing
& '.venv\Scripts\python.exe' 'tools/run_final_answer_evaluation.py' --mode dry-run --protocol 'output/final_evaluation/FINAL_EVALUATION_PROTOCOL.json' --manifest 'output/final_evaluation/SMOKE_TEST_MANIFEST.json' --checkpoint-dir 'output/final_evaluation/dry_run_checkpoint'
& '.venv\Scripts\python.exe' -m pytest -q
```

The freeze command intentionally refuses to overwrite frozen files. The paid command is documented in `AUTHORIZATION_GATE.md` and must not run until the separate authorization file exists after explicit user approval.
