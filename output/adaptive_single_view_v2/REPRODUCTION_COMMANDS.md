# Adaptive Single-View v2 reproduction commands

Run from the repository root with the checked-in local environment and local `all-MiniLM-L6-v2` snapshot.

```powershell
& '.venv\Scripts\python.exe' 'tools/analyze_screening_v3_corrected.py'
& '.venv\Scripts\python.exe' 'tools/freeze_adaptive_single_view_v2_protocol.py'
& '.venv\Scripts\python.exe' 'tools/amend_adaptive_single_view_v2_protocol_001.py'
& '.venv\Scripts\python.exe' 'tools/run_adaptive_single_view_v2_recalibration.py'
& '.venv\Scripts\python.exe' 'tools/analyze_adaptive_single_view_v2_recalibration.py'
& '.venv\Scripts\python.exe' 'tools/build_adaptive_single_view_v2_status_pdf.py'
& '.venv\Scripts\python.exe' -m pytest -q
```

The freeze and amendment commands are creation steps and intentionally refuse to overwrite already-frozen artifacts. For an existing completed checkout, begin with the recalibration command or validate with the tests. The workflow is retrieval-only and contains no paid or LLM calls.

