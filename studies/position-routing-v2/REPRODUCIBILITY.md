# Reproducing the AI-08 causal follow-up

The completed evidence is sealed. Use a fresh output location for analysis, and a new sibling study directory for full model replication. Do not overwrite a completed run, modify its freeze records or delete its outputs to repeat a measurement.

## Read-only verification

From the `<portfolio>` directory:

```bash
PYTHONDONTWRITEBYTECODE=1 <environment>/.venv/bin/python -B <project>/studies/position-routing-v2/code/verify_study.py
```

This verifies every file in the new manifest and the previous fixed-signal manifest, all frozen scientific files, the original AI-08 preparation and referenced upstream files, shared model assets, and closure accounting. It writes no files and performs no inference. A passed integrity check includes the accurately recorded failure of the strong scientific criterion; it is not a claim that all scientific gates passed.

## Replay saved-data analysis without a GPU

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 <environment>/.venv/bin/python -B - <<'PY'
import sys, tempfile
from pathlib import Path
study = Path('<project>/studies/position-routing-v2').resolve()
sys.path.insert(0, str(study / 'code'))
from analyze_causal import analyze
output = Path(tempfile.mkdtemp(prefix='ai08-causal-independent-'))
for bank in ['development', 'confirmation']:
    analyze(study / 'runs' / bank, output / bank)
print(output)
PY
```

The performed [analysis replay](analysis/ANALYSIS-REPLAY.json) used a fresh temporary directory and checked exact JSON and array equality for both banks. NPZ array contents define equality; ZIP timestamps need not match. The independent [evidence audit](analysis/EVIDENCE-AUDIT.json) separately reconstructed scores, intervention paths and bootstrap intervals. Completed audit scripts intentionally refuse to overwrite their saved audit result; use a new replica to repeat those audit commands.

## Environment and full model replication

The measured environment was Python 3.12, torch 2.13.0, transformers 5.16.1, NumPy 2.5.2 and SciPy 1.18.1 on a [6 GB NVIDIA GPU]. Inference used float32, eager attention, deterministic operations and two CPU threads. The existing model is `<models>/gpt2-medium`; all eight asset hashes are in [MODEL-ASSETS.json](provenance/MODEL-ASSETS.json). No network downloads or training are needed.

Copy this study's `code/`, `data/`, `protocol/` and `provenance/` to a **new sibling directory under AI-08's `studies/`**, with empty `runs/` and `analysis/` folders. Do not copy the old raw runs or final evidence manifest. The adapter resolves the read-only shared model and GPU lock relative to this directory. The old freeze files are historical records of the original design, not a new preregistration. Record new execution timestamps and explicitly identify the replica; a new qualification summary will have a new hash, so record a new replica freeze before its panels rather than passing it off as the original freeze.

Set `PYTHONDONTWRITEBYTECODE=1`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and `TOKENIZERS_PARALLELISM=false`. The execution order is:

1. Run `qualify.py`; require the declared numerical checks to pass.
2. Run `run_causal.py --split development --output NEW_STUDY/runs/development`.
3. Run `analyze_causal.py --run NEW_STUDY/runs/development --output NEW_STUDY/analysis/development` as a descriptive check. No head or threshold selection follows it.
4. Run `run_causal.py --split confirmation --output NEW_STUDY/runs/confirmation` and the matching analysis command with `confirmation` paths.
5. Run the independent input/evidence audits, saved-analysis replay and `replay_causal.py` for the fixed model-replay sample. Retain failures and diagnose them without relaxing tolerances or changing scientific thresholds.

Model measurement requires CUDA access. Before allocation the adapter opens the existing shared GPU lock read-only and takes an exclusive nonblocking lock. It requires at least 3 GiB available host memory, 4 GiB free GPU memory and 4 GiB free disk; do not terminate other work or replace the lock to bypass these checks. The entire retained run needs substantially more disk than that per-step minimum. Reserve at least 12 GB additional disk for a full replica with analysis and margin for temporary files.

Each recipient probe always executes eight copies of the same token sequence; unused condition slots are padded. Source extraction also uses batches of eight. This batch shape is part of the demonstrated numerical reproduction. The [performed replay](runs/replay/summary.json) repeated 32 fixed corner recipients in each bank, all 45 conditions each, and all 96 source extractions. Every compared array was bitwise identical. Different hardware, software or batch shapes require their own qualification; the earlier study's singleton tolerance failure is not erased by this result.

## Scientific implementation

`prepare_inputs.py` records tokenization-only selection, semantic positions, name/source pairing, controls and rotations. `core.py` captures the twelve source heads and adds the cached correction at the repeated-subject position. `mechanism.py` implements final-position Q/K/V/P/Z replacement and query-only conditions while restoring the underlying attention function after each call. `run_causal.py` saves the complete 45-condition panel, full-vocabulary logits, selected-layer observations, baseline keys/values and hashes.

`analyze_causal.py` computes the fixed order contrasts and aggregate ratios with crossed name/context resampling. `test_analysis.py` checks ratio handling and that repeated within-unit rows do not manufacture precision. `audit_inputs.py`, `audit_causal.py`, `replay_analysis.py` and `replay_causal.py` provide the independent checks. `make_figures.py` uses only the saved confirmation summary. `summarize_study.py` accounts for all 40,256 physical model-example evaluations, and `seal_study.py` creates a final manifest once.

The adapter configuration retains unused fields inherited from V1, including `strengths` and `random_replicates`; they do not enumerate this experiment's conditions. The frozen `condition_plan` and saved `conditions.json` records define this follow-up's 45 conditions at source strength +1. The copied `response_models.py` is unused in this causal analysis. No fitting or whole-model head search is part of the primary analysis.
