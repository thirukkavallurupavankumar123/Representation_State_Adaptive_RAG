# Exact Reproduction Commands

Every command below was executed during this audit unless explicitly marked
**NOT RUN**. All are PowerShell, from the repository root. None makes a paid API
call except those in §7, which were **not run**.

```powershell
Set-Location 'c:\Users\pavan kumar\OneDrive\Documents\research_projectt'
$env:PYTHONPATH = (Resolve-Path -LiteralPath 'src').Path
```

## 1. Environment and integrity

```powershell
git status --short
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pip install pytest reportlab pymupdf   # added by this audit
```

Verify the embedding snapshot against recorded provenance:

```powershell
.\.venv\Scripts\python.exe -c "import hashlib,json,pathlib; p=pathlib.Path('models/all-MiniLM-L6-v2/model.safetensors'); print(hashlib.sha256(p.read_bytes()).hexdigest() == json.load(open('MODEL_PROVENANCE.json'))['weight_sha256'])"
```

Confirm `.env` is untracked and no key is committed:

```powershell
git ls-files --error-unmatch .env      # must fail
git ls-files | %{ Select-String -Path $_ -Pattern 'gsk_[A-Za-z0-9]{20,}' -ErrorAction SilentlyContinue }
```

## 2. Test suite (73 tests, project venv)

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest -v -p no:warnings   # produces TEST_REPORT.txt content
```

Individual contract suites:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_primary_pipeline_contracts.py -q   # 34
.\.venv\Scripts\python.exe -m pytest tests/test_experiment_contracts.py -q         # 10
.\.venv\Scripts\python.exe -m pytest tests/test_defect_regressions.py -q           #  6
```

## 3. Calibration reproduction (no Groq calls)

Version 1 — against the **unmodified** code, before any remediation:

```powershell
.\.venv\Scripts\python.exe -m sar_rag.calibration `
  --qasper-train .\datasets\raw\qasper\data\qasper-train-v0.3.json `
  --musique-train .\datasets\raw\musique\data\musique_full_v1.0_train.jsonl `
  --output .\output\independent_verification\calibration_v1_reproduction.json `
  --per-stratum 5 --search-trials 32
```

Version 2 — after defect remediation:

```powershell
.\.venv\Scripts\python.exe -m sar_rag.calibration `
  --qasper-train .\datasets\raw\qasper\data\qasper-train-v0.3.json `
  --musique-train .\datasets\raw\musique\data\musique_full_v1.0_train.jsonl `
  --output .\output\independent_verification\calibration_v2_manifest.json `
  --per-stratum 5 --search-trials 32
```

`output\calibration\frozen_manifest.json` was **never** written to. Result:
v0 ≡ v1 ≡ v2 on all 17 compared fields; zero thresholds changed. See
`VERSION1_VS_VERSION2_CALIBRATION.json`.

## 4. Leakage audit (no Groq calls)

The shipped module reproduces the published numbers:

```powershell
.\.venv\Scripts\python.exe -m sar_rag.leakage_audit `
  --qasper-train .\datasets\raw\qasper\data\qasper-train-v0.3.json `
  --qasper-test  .\datasets\raw\qasper\data\qasper-test-v0.3.json `
  --musique-train .\datasets\raw\musique\data\musique_full_v1.0_train.jsonl `
  --musique-dev   .\datasets\raw\musique\data\musique_full_v1.0_dev.jsonl `
  --musique-registry .\datasets\raw\musique\data\dev_test_singlehop_questions_v1.0.json `
  --output .\output\independent_verification\leakage_audit_reproduced.json
```

The **calibration-isolation** audit (the check that did not previously exist)
is generated inline and written to `DATASET_LEAKAGE_AUDIT.json`. It compares the
60 fitted calibration examples against both evaluation splits.

## 5. Retrieval-only ablation screening (no Groq calls)

```powershell
.\.venv\Scripts\python.exe tools\retrieval_only_screening.py `
  --per-stratum 5 --seed 20260804
```

Seed `20260804` is deliberately different from the calibration seed `20260803`,
so screening does not re-use the examples that selected the thresholds. Touches
**no evaluation split**. Produces `RETRIEVAL_ONLY_ABLATIONS.csv`,
`REPRESENTATION_SELECTION_DIAGNOSTICS.json` and
`REPRESENTATION_ROUTING_PER_EXAMPLE.csv`.

## 6. Report and PDF

```powershell
.\.venv\Scripts\python.exe tools\build_audit_pdf.py
.\.venv\Scripts\python.exe -c "import pymupdf,pathlib; d=pymupdf.open('output/independent_verification/SAR_RAG_INDEPENDENT_VERIFICATION.pdf'); out=pathlib.Path('output/independent_verification/pdf_page_renders'); out.mkdir(exist_ok=True); [p.get_pixmap(dpi=110).save(str(out/f'page_{i:02d}.png')) for i,p in enumerate(d,1)]"
```

Demonstrate the missing-table fix without touching the version-1 analysis:

```powershell
.\.venv\Scripts\python.exe -m sar_rag.q2_analysis `
  --results .\output\pilot_fair_generator_qasper2\results.json `
  --dataset-input .\datasets\normalized\qasper_validation_sample20.jsonl `
  --output-dir .\output\independent_verification\q2_analysis_pilot_regenerated `
  --resamples 2000 --failure-sample 100
```

`table_failures.csv` and `manual_failure_review_sample.csv` are now created as
header-only files instead of being silently skipped.

## 7. Confirmatory evaluation — **NOT RUN**

**Do not execute until every gate in `EXACT_PAID_EXECUTION_PLAN.md` §6 is
satisfied and the project owner explicitly authorizes the spend.** Approximately
18,855 API calls, ~63M prompt and ~10M completion tokens, ~56 h serial.

```powershell
$env:GROQ_INPUT_COST_PER_MILLION        = '<verified run-date rate>'
$env:GROQ_CACHED_INPUT_COST_PER_MILLION = '<verified run-date rate>'
$env:GROQ_OUTPUT_COST_PER_MILLION       = '<verified run-date rate>'

.\.venv\Scripts\python.exe -m sar_rag.experiments `
  --input .\datasets\normalized\qasper_test.jsonl `
  --output-dir .\output\independent_verification\confirmatory\qasper `
  --calibration-manifest .\output\calibration\frozen_manifest.json

.\.venv\Scripts\python.exe -m sar_rag.experiments `
  --input .\datasets\normalized\musique_full_validation.jsonl `
  --output-dir .\output\independent_verification\confirmatory\musique `
  --calibration-manifest .\output\calibration\frozen_manifest.json
```

Resume with `--resume` appended to the **identical** command. Never pass
`--limit` or `--static-k`; confirmatory mode rejects both.

Then:

```powershell
.\.venv\Scripts\python.exe -m sar_rag.q2_analysis `
  --results .\output\independent_verification\confirmatory\qasper\results.json `
  --results .\output\independent_verification\confirmatory\musique\results.json `
  --dataset-input .\datasets\normalized\qasper_test.jsonl `
  --dataset-input .\datasets\normalized\musique_full_validation.jsonl `
  --output-dir .\output\independent_verification\q2_analysis `
  --resamples 10000 --failure-sample 100
```

## 8. Files changed by this audit

Modified source (6): `evaluation.py`, `traditional.py`, `calibration.py`,
`experiments.py`, `q2_analysis.py`, `representations.py`.
Modified docs (2): `CALIBRATION_AUDIT.md`, `ONE_CALL_RESEARCH_PIPELINE.md`.
New tests (3): `test_primary_pipeline_contracts.py`, `test_experiment_contracts.py`,
`test_defect_regressions.py`.
New tools (2): `retrieval_only_screening.py`, `build_audit_pdf.py`.

**Not modified:** `output/calibration/frozen_manifest.json`,
`output/pilot_fair_generator_qasper2/`, `output/pilot_frozen_qasper2/`,
`output/q2_analysis_pilot/`, `output/pdf/`. All version-1 historical artifacts
are preserved byte-for-byte.
