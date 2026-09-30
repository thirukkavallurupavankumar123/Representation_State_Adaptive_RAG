# Reproduction commands: local_qwen_4bit_reduced_v1

Run everything from the repository root inside WSL2 Ubuntu 24.04, using the local Qwen environment:

```bash
cd /mnt/f/research/research_projectt
PY=/home/gcet/sar_rag_qwen_env/bin/python
git checkout local-qwen-reduced-eval
```

The model is expected at `/mnt/f/research/models/Qwen3-8B` (Qwen/Qwen3-8B @ `b968826d9c46dd6066d109eabc6255188de91218`).
Pass `--model-path` to `smoke` or `generate` to use another location. No command calls a remote inference API.

## 0. Tests and integrity (no GPU, no gold)

```bash
$PY -m unittest tests.test_local_qwen_4bit -v
$PY tools/local_qwen_4bit.py fetch-inputs        # Git-LFS sealed requests/audit -> output/local_qwen_4bit/inputs/lfs (SHA-256 verified)
$PY tools/local_qwen_4bit.py verify-integrity    # prior frozen artifacts (CRLF-equivalent hashes, object fingerprints)
python3 output/local_qwen_4bit/environment/verify_weight_hashes.py   # after sha256sum, see environment/qwen3_8b_local_sha256.txt
```

## 1. Development smoke (design-exposed smoke-v1.1 pairs only)

```bash
$PY tools/local_qwen_4bit.py smoke --label r1          # 8 questions x 3 systems; starts the budget clock on first use
$PY tools/local_qwen_4bit.py smoke --label r1_repeat   # repeatability diagnostic (same seeds)
python3 output/local_qwen_4bit/smoke/compare_repeat.py r1 r1_repeat
```

## 2. Freeze (runtime-derived N, protocol, sample manifest)

```bash
$PY tools/local_qwen_4bit.py freeze --accepted-smoke r1
```

This command is create-once. It writes `protocol/PROTOCOL.json` (+ `.sha256`), `manifests/SAMPLE_MANIFEST.json` (+ `.sha256`) and `manifests/FREEZE_RECORD.json`.
N comes from the measured smoke throughput and the remaining budget, which is recorded in `protocol.runtime_estimate`. Re-running `freeze` on the same data at a different time can therefore produce a different N. To reproduce this study's sample exactly, call `build_sample(N)` with the frozen N.

## 3. Reduced generation (no gold access)

```bash
$PY tools/local_qwen_4bit.py generate   # resumable: rerun the identical command after any interruption
```

## 4. Seal predictions (before any gold access)

```bash
$PY tools/local_qwen_4bit.py seal
```

## 5. Gold acquisition and scoring (only after the seal)

```bash
$PY tools/score_local_qwen_4bit.py fetch-gold   # official archives, SHA-256 checked; normalized files must match the pool's recorded hashes
$PY tools/score_local_qwen_4bit.py score
```

## 6. Report

See `reports/FINAL_REPORT.md`. The report was assembled from `metrics/RESULTS.json`, `comparisons/PAIRED_COMPARISONS.csv`, `predictions/PREDICTION_SEAL.json` and the run logs.
