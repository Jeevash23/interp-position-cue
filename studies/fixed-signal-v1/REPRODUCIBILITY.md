# Reproducing AI-08

The completed evidence is immutable. Use a new output directory for additional analyses and a new sibling study directory for a full model rerun. The measurement scripts refuse an existing run directory. Do not delete completed outputs to make a rerun fit.

## Verify saved evidence

From the `<portfolio>` field directory:

```bash
PYTHONDONTWRITEBYTECODE=1 <environment>/.venv/bin/python -B <project>/studies/fixed-signal-v1/code/verify_study.py
```

This checks every completed-study file against `EVIDENCE-MANIFEST.json`, frozen code/model-selection inputs, original preparation files, referenced upstream files, model assets and the final accounting. It deliberately expects the singleton replay's recorded failed tolerance; integrity verification is not a claim that every scientific or numerical gate passed. It performs no model inference, writes no files and can be repeated.

## Regenerate the primary analysis without a GPU

Use the recorded environment and constrain numerical libraries to two threads. The following reproduces the development selection and evaluates both held-out banks into a fresh temporary directory, without modifying the evidence:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 <environment>/.venv/bin/python -B - <<'PY'
import sys, tempfile
from pathlib import Path
study = Path('<project>/studies/fixed-signal-v1').resolve()
sys.path.insert(0, str(study / 'code'))
from analyze_responses import analyze
output = Path(tempfile.mkdtemp(prefix='ai08-independent-analysis-'))
analyze(study / 'runs/development', output / 'development')
analyze(study / 'runs/holdout', output / 'holdout', output / 'development')
analyze(study / 'runs/confirmation', output / 'confirmation', output / 'development')
print(output)
PY
```

The performed [analysis replay](analysis/ANALYSIS-REPLAY.json) reproduced all 18 JSON/numerical artifacts exactly, including cross-validation selection. Array contents, rather than ZIP timestamps in NPZ containers, define equality.

## Model environment and measurement order

The measured environment was Python 3.12, torch 2.13.0, transformers 5.16.1, NumPy 2.5.2 and SciPy 1.18.1. The existing model is at `<models>/gpt2-medium`; its eight asset fingerprints are in `provenance/MODEL-ASSETS.json`. The adapter uses Hugging Face directly, float32, eager attention, deterministic operations and two CPU threads. It does not import an older model runner or require TransformerLens at execution time.

For a full independent rerun, copy this study's `code/`, `protocol/`, `data/` and `provenance/` into a **new sibling directory under AI-08's `studies/`**, create empty `runs/` and `analysis/` directories there, and retain the shared model/environment as read-only. Do not copy old `runs/`, `analysis/` or the completed evidence manifest. The adapter resolves the shared model relative to that study location. The copied files retain the original design; record new run timestamps and a new final manifest rather than treating the original freeze as a new preregistration.

Run, in order, with the new study path:

1. `qualify.py`; inspect numerical and clean-competence results before proceeding.
2. `run_responses.py --split development --output NEW_STUDY/runs/development`.
3. `analyze_responses.py --run NEW_STUDY/runs/development --output NEW_STUDY/analysis/development`; freeze the selected model for that rerun.
4. `run_responses.py --split holdout --output NEW_STUDY/runs/holdout` and analyze it with `--fit NEW_STUDY/analysis/development`.
5. `run_confirmation.py --split confirmation --output NEW_STUDY/runs/confirmation` and analyze it using that same frozen development model.
6. Run the evidence, input and saved-analysis audits. For numerical replay, retain both singleton and original-batch results and quantify any discrepancy before interpreting it.

Set `PYTHONDONTWRITEBYTECODE=1`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and `TOKENIZERS_PARALLELISM=false`. Model execution requires access to the local CUDA device. The adapter opens the existing shared GPU lock read-only and acquires an exclusive nonblocking lock before allocation. If the lock or resource guards fail, leave other work intact. It requires at least 3 GiB available host memory, 4 GiB disk and 4 GiB free GPU memory at startup. This completed evidence occupies about 5.44 GB, so reserve substantially more than the minimum disk guard for an entire new run.

## Files that implement the scientific design

`core.py` handles semantic-position capture and additive interventions. `run_responses.py` and `run_confirmation.py` use cached source corrections and save the full condition panels. `response_models.py` implements gain/rank selection and crossed inference. `analyze_responses.py` generates the primary results without holdout fitting. `test_models.py` contains the passed synthetic gain/rotation and crossed-fold tests. `describe_controls.py` produces secondary slices and dose descriptions; these were not used to change the primary model or decision.

`audit_evidence.py` independently reconstructs saved scores and bootstrap intervals. `audit_inputs.py` retokenizes all inputs and checks bank separation/history. `replay_analysis.py` documents the performed full analysis replay. `replay_model.py` uses singleton execution on fixed corner blocks. `investigate_replay.py` repeats original batch shapes after the singleton tolerance failure; `replay_sensitivity.py` measures its score consequences. `summarize_study.py` records the exact inference counts and scoped decision. None of these scripts publishes or transmits data.

## Numerical reproducibility boundary

All original-batch checks were bitwise identical. The singleton run changed no top-1 answer and altered the sampled model-comparison improvement by at most 0.000000255, but exceeded the fixed 0.0005 absolute state-coordinate tolerance. Its failed status is part of the evidence. The original batching is required for the demonstrated exact replay; a different library version, accelerator or batch shape should be independently qualified. The completed study did not demonstrate a routing mediator.
