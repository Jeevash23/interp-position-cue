# Reproducing the AI-08 pathway and interaction study

This study separates selection on existing development inputs from testing on fresh names, contexts and wording. The completed evidence must remain unchanged. Use a fresh output directory for saved-data analysis and a new sibling study directory for model replication.

## Verify evidence without inference

From the `<portfolio>` directory:

```bash
PYTHONDONTWRITEBYTECODE=1 <environment>/.venv/bin/python -B <project>/studies/pathway-interactions-v3/code/verify_study.py
```

The verifier checks every sealed V3 file, all scientific freezes and closure accounting, then runs the V2 verifier to check both earlier studies, original preparation records and model assets. It performs no inference and writes no files. A passed integrity check does not turn a failed scientific criterion into a success.

## Regenerate the saved-data analysis

The original frozen analysis computes the numerical results correctly, but its first JSON export encountered a NumPy boolean that the standard JSON encoder could not serialize. The original code, failed log and numerical output remain preserved. A small wrapper converts NumPy scalars to equivalent Python scalars solely at the write boundary. The paired numerical arrays before and after this repair are bitwise identical. See [repair provenance](provenance/SERIALIZATION-REPAIR.json) and [equality check](analysis/SERIALIZATION-EQUALITY.json).

Use the wrapper for repeated analysis in fresh directories:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 <environment>/.venv/bin/python -B - <<'PY'
import sys, tempfile
from pathlib import Path
study = Path('<project>/studies/pathway-interactions-v3').resolve()
sys.path.insert(0, str(study / 'code'))
import core
from serialize_results import native
from analyze import analyze
original_write = core.write
core.write = lambda path, value: original_write(path, native(value))
output = Path(tempfile.mkdtemp(prefix='ai08-pathway-analysis-'))
try:
    for bank in ['development', 'confirmation']:
        analyze(study / 'runs' / bank, output / bank)
finally:
    core.write = original_write
print(output)
PY
```

The [saved-analysis replay](analysis/ANALYSIS-REPLAY.json) compares all numerical arrays and JSON results exactly. It uses array contents rather than NPZ archive timestamps. The [independent evidence audit](analysis/EVIDENCE-AUDIT.json) reconstructs the primary scores, selection, signals and crossed-bootstrap intervals through separate code.

## Evidence coverage

Every condition retains the two original float32 target-name logits and the final residual vector. Full-vocabulary top choices, log normalizers, entropy and KL were computed before other logits were discarded. All five natural states per recipient retain full-vocabulary logits and Q, head-output and MLP-output caches.

Full condition logits and complete intervention traces are retained on the sample fixed before measurements: first/last name units crossed with first/last contexts, role zero and order zero, both templates and both corruptions. This gives four screen panels and 16 panels in each factorial bank. The screen's four inputs also occur in the development bank, with different condition panels. The audit independently checks primary margins for all 128,448 condition records; independent full-vocabulary and site-replacement checks cover the declared 5,048 full-trace records. The scope of the audit is not all-vocabulary reconstruction for every condition.

## Full model replication

The measured environment uses Python 3.12, torch 2.13.0, transformers 5.16.1, NumPy 2.5.2 and SciPy 1.18.1. Inference uses the existing offline GPT-2 Medium checkpoint in float32, eager attention, deterministic operations and two CPU threads. The GPU is a [6 GB NVIDIA GPU]. Model assets and fingerprints are recorded in [MODEL-ASSETS.json](provenance/MODEL-ASSETS.json). No training or model download is required.

Create a new sibling directory under AI-08's `studies/`. Copy `code/`, `data/`, `protocol/config.json` and `protocol/PLAN.md`, and keep the old freeze files as historical references under the new replica's provenance directory. Create empty `runs/` and `analysis/` directories. Do not copy completed outputs or the completed manifest. The adapter resolves the shared model and GPU lock relative to the sibling study location. Record new freeze files at the appropriate stages; original run timestamps and output hashes must not be represented as records of the replica.

Set `PYTHONDONTWRITEBYTECODE=1`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `TOKENIZERS_PARALLELISM=false`, `OPENBLAS_NUM_THREADS=2`, `OMP_NUM_THREADS=2` and `MKL_NUM_THREADS=2`. Run the following stages in order, using the new study paths:

1. `test_statistics.py`, input auditing, then the design freeze and `qualify.py`. Require numerical qualification before further model work.
2. `run_study.py --phase screen --output NEW_STUDY/runs/screen` and `select.py`. Preserve the complete ranking and freeze the selected two heads, one MLP and sham controls.
3. `run_study.py --phase development --output NEW_STUDY/runs/development`, then `serialize_results.py --module analyze --run NEW_STUDY/runs/development --output NEW_STUDY/analysis/development`.
4. Freeze development expectations, the unchanged confirmation criteria and verification procedures. Run `run_study.py --phase confirmation --output NEW_STUDY/runs/confirmation`, then its matching wrapped analysis.
5. Run `audit_evidence.py`, `serialize_results.py --module replay_analysis`, and `replay_model.py`. Each refuses to overwrite completed output. Retain failures and diagnose them rather than deleting or relaxing criteria.

Model execution requires CUDA access. The adapter opens the existing shared GPU lock read-only and acquires an exclusive nonblocking lock before allocation. It requires at least 3 GiB available host memory, 4 GiB free GPU memory and 4 GiB free disk. Reserve at least 10 GB additional disk for a complete new replica; the disk guard is a per-step minimum, not a total output estimate. Leave other work intact if the shared lock or resource guard is busy.

Every recipient probe uses eight copies of the same prompt; padding is counted as inference work. The numerical reproduction claim is specific to this batch shape and recorded environment. Other batch shapes, devices and software versions require qualification. The complete final-position attention/MLP clamps and transplants are boundary controls; their success does not identify a compact mechanism.

## Design and code map

`prepare_inputs.py` fixes tokenization, source pairing, corruption variants and fresh confirmation inputs. `core.py` is the preserved base adapter. `path_engine.py` implements the Q/A/M interventions and enumerates the fixed screen and factorial conditions. `run_study.py` measures the panels and retains compact primary evidence plus the fixed full-trace sample. `select.py` uses only screen data and never reads confirmation results.

`analyze.py` contains the frozen crossed inference and scientific gates. `serialize_results.py` repairs only serialization. `audit_inputs.py` checks input alignment and separation; `audit_evidence.py` independently reconstructs selection, all primary effects and the declared traces. `replay_analysis.py` and `replay_model.py` reproduce saved analyses and model evaluations. `summarize_study.py` compares frozen development expectations with confirmation and records exact accounting. `make_figures.py` plots only saved confirmation results.

The protocol fixes four primary quantities in each of four corruption/strength strata, giving 16 simultaneous intervals at 99.6875%. Secondary interaction and specificity intervals are descriptive 95%. Ratios refer to different interventions and are not additive shares of the natural computation. The [completed report](report/AI-08-PATHWAY-FOLLOWUP.md) records both positive findings and criteria that failed.
