# Representation-State Adaptive RAG

This repository implements the research pipeline without a fixed final retrieval `k`.
The only small fixed depth is `probe_depth` (default 3), which diagnoses which chunk
representations are useful. Retrieval then continues in batches while deterministic
similarity, novelty and marginal-utility rules add useful context. Deduplication happens
only after adaptive retrieval stops.

The method is **model-fine-tuning-free, not supervision-free**: official training-set
evidence labels calibrate its deterministic controls and the baseline's `k`. No neural
weights are trained, and validation/test labels never select settings.

## Fixed model choices

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Final answering only: Groq `openai/gpt-oss-120b`

The primary method makes exactly one online LLM call: the final evidence-grounded
answer/abstention call. Query embedding, probing, RRF, representation selection,
adaptive retrieval, stopping and deduplication are mathematical/local operations.

The checkpoint is loaded directly with Transformers/PyTorch and uses attention-mask mean
pooling followed by L2 normalization, matching its Sentence Transformer configuration.
`HashingEncoder` and scripted reasoning classes exist only for deterministic unit tests.
They are not used by `SarRagSystem.research` and must not be used to report research results.

## Pipeline

1. Parse sources into stable canonical units and preserve document/version provenance.
2. Build sentence-window, paragraph, semantic, and hierarchical views.
3. Retrieve the top `probe_depth` from every view.
4. Align overlapping candidates through canonical source-unit overlap and compute
   normalized reciprocal-rank-fusion scores without an LLM.
5. Select the first representation by total normalized-RRF coverage. Select a second
   only when it adds the configured minimum marginal RRF coverage.
6. Reuse selected probe candidates, then retrieve further batches from selected indexes.
7. Score each candidate using cosine relevance, canonical novelty, semantic redundancy
   and marginal utility. Pause a view after configured no-gain patience or a supporting
   similarity-gap signal. Token, candidate, round and chunk limits are safety caps.
8. Perform exact, containment and conservative near-duplicate removal after retrieval.
9. Compact evidence within the tokenizer-token context budget.
10. Make one Groq call for an evidence-only answer or `INSUFFICIENT`, then validate citations.

The returned `final_k` is simply the number of evidence items remaining after those steps.
It is an output, never a retrieval target.

The previous per-candidate LLM-judged controller remains available only as
`LlmJudgedSarRagSystem` for a historical ablation. It is not the primary method.

## Run

Create an isolated environment, install the locked dependencies and project, then pass
one or more Markdown/plain-text sources. The CLI automatically reads `GROQ_API_KEY`
and `GROQ_MODEL` from `.env`; do not commit that file.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps --no-build-isolation
Copy-Item .\.env.example .\.env
# Edit .env and replace the placeholder key.
.\.venv\Scripts\python.exe -m sar_rag --document .\policy.md `
  --query "Who is eligible and under what conditions?" --output .\run.json
```

For exact repeatability, pass a resolved Hugging Face commit with
`--embedding-revision`. Every JSON run contains source/version IDs, representation
selection diagnostics, the adaptive-retrieval trace, final evidence, and the emergent
`final_k`.

When `models/all-MiniLM-L6-v2` exists, production runs load that verified local Hugging
Face snapshot; otherwise they resolve the official model ID. `MODEL_PROVENANCE.json`
records the downloaded snapshot and weight checksum.

Run normalized development/test JSONL files with `sar-rag-benchmark`; the exact input
contract is shown in `benchmark_schema.example.jsonl`. Dataset conversion must preserve
the source text and gold spans verbatim. Only the verified QASPER v0.3 and MuSiQue-Full
v1.0 schemas have adapters; unknown schema variants fail explicitly.

## Training-only calibration

Run calibration before validation or test evaluation. It makes no Groq calls, samples
only official training splits, searches the registered mathematical thresholds, uses a
held-back training selection partition, calibrates the baseline `k`, and writes dataset
checksums plus the selected configuration to a manifest:

```powershell
.\.venv\Scripts\python.exe -m sar_rag.calibration `
  --qasper-train .\datasets\raw\qasper\data\qasper-train-v0.3.json `
  --musique-train .\datasets\raw\musique\data\musique_full_v1.0_train.jsonl `
  --output .\output\calibration\frozen_manifest.json `
  --per-stratum 5 --search-trials 32
```

Use `--smoke-limit` only to verify code. Such output is marked `smoke_not_frozen` and
cannot be used in a paper. The proposed system's final `k` is never calibrated.

Run the paired confirmatory comparison only with that frozen manifest:

```powershell
.\.venv\Scripts\python.exe -m sar_rag.experiments `
  --input .\datasets\normalized\qasper_validation.jsonl `
  --output-dir .\output\confirmatory\qasper `
  --calibration-manifest .\output\calibration\frozen_manifest.json
```

Confirmatory mode rejects missing/smoke manifests, baseline-`k` overrides, and `--limit`.
Use `--pilot --limit N` for engineering checks only; pilot results cannot support claims.
If an API or machine failure interrupts a run, repeat the identical command with `--resume`.
Resume is refused unless the input, calibration manifest, model, controls, and evaluator
condition match the saved fingerprint exactly.

Confirmatory runs also require frozen Groq pricing through
`GROQ_INPUT_COST_PER_MILLION`, `GROQ_CACHED_INPUT_COST_PER_MILLION`, and
`GROQ_OUTPUT_COST_PER_MILLION` (USD). Record rates verified on the run date; the code
does not silently assume that provider pricing has remained unchanged.

## Downloaded research datasets

The implementation now includes schema-validated adapters for the official releases:

- `datasets/raw/qasper/data`: QASPER v0.3 train/dev/test papers and annotations.
- `datasets/raw/musique/data`: MuSiQue-Full v1.0 train/dev/test and the official
  dev/test seed-question leakage registry.
- `datasets/DATASET_MANIFEST.json`: sources, licenses, split counts and SHA-256 hashes.

QASPER is the primary representation-granularity dataset. MuSiQue-Full is the primary
multi-hop and insufficiency dataset. HotpotQA is reserved for later external
generalization because it substantially overlaps MuSiQue's multi-hop purpose; QuALITY
is not part of the initial experiment. The custom representation-state benchmark must
be authored and audited - it cannot be downloaded as an existing public dataset.

Prepare deterministic normalized files without inventing a `policy.md`:

```powershell
python -m sar_rag.dataset_adapters --dataset qasper --split validation `
  --output .\datasets\normalized\qasper_validation.jsonl

python -m sar_rag.dataset_adapters --dataset musique_full --split validation `
  --output .\datasets\normalized\musique_full_validation.jsonl
```

Use `--limit 20` only for a smoke run; first-N is not a representative calibration
sample. Do not calibrate on validation or test. Freeze probe depth, RRF constant,
batch size, budgets and stopping controls using training data only. Then use QASPER
development for validation and QASPER test for the final held-out report. Use
MuSiQue-Full development for the locally scored final report because MuSiQue's official test
labels are hidden, so local answer/evidence scoring uses validation; test predictions
are for official submission only.

Then run a small, cost-controlled benchmark after setting `GROQ_API_KEY`:

```powershell
python -m sar_rag.benchmark `
  --input .\datasets\normalized\qasper_validation_sample20.jsonl `
  --output .\output\qasper_sample20_results.json `
  --limit 2
```

## Verification

```powershell
python -m unittest discover -s tests -v
```

See `UPDATED_RESEARCH_SPECIFICATION.md`, `DETAILED_IMPLEMENTATION_BLUEPRINT.md`, and
`DATASET_SELECTION_DECISION.md` for the research protocol and evaluation design.

## Q2 journal evaluation package

The evaluator now records complete-evidence recovery, token precision/recall/F1,
P/R@1/3/5/10/20, MRR, nDCG, answerability errors, paired bootstrap/randomization,
paired effect size, exact McNemar tests, Holm correction, efficiency distributions,
index size, embedding work, API attempts, representation states, adaptive-k behavior,
subgroups, deterministic failure samples, and registered ablations.

Run one registered condition by adding, for example, `--condition no_novelty` to
`python -m sar_rag.experiments`. Available names are defined in
`src/sar_rag/ablations.py`; every condition is stored in the immutable run fingerprint.
Use a separate output directory per dataset and condition.

After the completed result files exist, generate publication tables and figures with:

```powershell
.\.venv\Scripts\python.exe -m sar_rag.q2_analysis `
  --results .\output\confirmatory\qasper\results.json `
  --results .\output\confirmatory\musique\results.json `
  --dataset-input .\datasets\normalized\qasper_test.jsonl `
  --dataset-input .\datasets\normalized\musique_full_validation.jsonl `
  --output-dir .\output\q2_confirmatory
```

`output/pdf/SAR_RAG_Q2_RESEARCH_EVALUATION.pdf` documents all implemented
calculations and marks unexecuted held-out and ablation results as pending.
