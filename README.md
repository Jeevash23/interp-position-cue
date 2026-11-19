# A fixed internal signal acts as a relative-position cue

Code and results for the preprint *A fixed internal signal acts as a relative-position cue: name order decides whether it helps or harms indirect-object identification in GPT-2 Medium*, by Richard Jeevash (link to be added when it is posted).

This repository holds the code, frozen inputs, study records and saved summaries behind it; the saved outputs and
the frozen protocols are attached to its v1.0 release (see [Release files](#release-files)). Every number,
generated table and plotted value in the manuscript can be recomputed from the saved outputs on a CPU in under a
minute, without a model or network.

## Release files

The saved outputs and the frozen protocols are not in this repository's files; they are attached to its v1.0 release:

| File | Size | Contents |
|---|---|---|
| `per-run-outputs.tar.gz` | 3.9 MB | The saved outputs: the 3 compressed archives in `data/saved-outputs/`, which `make unpack` places in the study folders and checks file by file |
| `frozen-protocols.tar.gz` | 10 kB | The studies' protocols, predictions, amendments, approvals and freeze records (16 files), at their places in the folder, where the checks read them |
| `SHA256SUMS` | | Checksums of the two archives |

Download the three files into an empty folder, check them there, and unpack them into this one:

```sh
sha256sum -c SHA256SUMS
tar -xzf per-run-outputs.tar.gz -C /path/to/interp-position-cue
tar -xzf frozen-protocols.tar.gz -C /path/to/interp-position-cue
```

After unpacking, `sha256sum -c SHA256SUMS` in this folder checks every file that its `SHA256SUMS` lists, the
unpacked attachments included, and the checks below run as described.

## What the paper asks

Interpretability experiments often add a signal cached from one prompt to another prompt, and the signal's
effect can change with the prompt that receives it. The paper asks why, in GPT-2 Medium on indirect-object
sentences ("In the library, Maya and Noah talked. Noah handed a note to" → Maya). The signal is the
clean-minus-corrupted output of twelve attention heads at the repeated name, cached once from a source sentence
and added to corrupted sentences with other wordings, places and name orders. Three studies, each designed from
the previous one's results and frozen before its confirmation run, test whether the signal carries the relative
position of the repeated name, whether that effect survives source sentences with other names, and which heads
and MLP blocks carry it.

## Studies

| Manuscript | Folder | What it did |
|---|---|---|
| Study 1 | `studies/fixed-signal-v1` | One gain per context against a directional response model; 1,408 recipients (1,152 in the confirmation set), 18,304 condition records |
| Study 2 | `studies/position-routing-v2` | Transfer across names and the queries of five name-mover heads; 640 recipients (512 confirmation), 28,800 records |
| Study 3 | `studies/pathway-interactions-v3` | Development screen of 284 components, two heads and one MLP block frozen, factorial confirmation on 1,024 new cases |

Each study folder keeps its code, protocol and freeze records (`protocol/`), frozen inputs (`data/`), the
preparation files it started from (`provenance/`), and a manifest of every file it produced
(`EVIDENCE-MANIFEST.json`). The folder names are the original ones, because the studies' verifiers refer to
each other by them. The freezes are local hash records written by the study code, not public preregistrations.

## What is in this folder

| Path | Contents |
|---|---|
| `studies/` | The three studies (above); after `make unpack`, also their saved analyses and run summaries |
| `paper/make_numbers.py`, `paper/make_figures.py` | Every number, generated table and figure of the manuscript from the saved outputs; `paper/generated/` and `paper/figures/figure-data.json` hold the values the manuscript was built with |
| `analysis/` | Post hoc analyses of saved outputs and the replay of each study's saved-output analysis; `analysis/results/` holds their outputs |
| `provenance/` | `verify_studies.py` (the wrapper that runs the three frozen verifiers), the record of the 9 October file move, project amendments, and the 25 preparation files of 7 October |
| `reference/` | The earlier experiment's configuration, from which these studies take their twelve source heads |
| `data/saved-outputs/` | Saved analyses and run summaries as compressed archives (attached to the release: `per-run-outputs.tar.gz`) |
| `tools/`, `models/` | Unpacking, freeze checking and model download helpers; the pinned checkpoint |
| `pyproject.toml`, `uv.lock` | The exact environment of the runs (see `ENVIRONMENT.md`) |
| `RELEASE-CHANGES.md`, `release-changes.json` | Every file that differs from the original project, with the original's SHA-256, and every frozen file not shipped |

## Install

```sh
make setup        # uv sync --frozen --extra experiment --group dev: creates .venv from uv.lock
```

This downloads about 3.0 GB of packages (most of it PyTorch's CUDA libraries); the checks below need only
NumPy, SciPy and Matplotlib, and `ENVIRONMENT.md` gives a small CPU-only route. Pass `PYTHON=/path/to/python` to
any `make` target to use another environment.

## Quick check (CPU, under a minute, no model, no network)

First unpack the two release attachments into this folder (see [Release files](#release-files)): the checks read
the saved outputs and the freeze records they contain.

```sh
make quick-check
```

It unpacks the saved outputs and compressed inputs (checking every SHA-256), checks every freeze record against
the files it names, and then recomputes every macro and generated table of the manuscript
(`paper/make_numbers.py --check`) and every plotted value (`paper/make_figures.py --check`) from the saved
outputs, comparing them with the shipped values. Expected end of the output:

```text
total: 8 changed for the release (documented), 2 not shipped (documented), 72 unchanged
ok  665 macros and 7 tables match the saved outputs
ok  figure data match the saved outputs (52 plotted values)
```

## More from the saved outputs

```sh
make numbers      # rewrite paper/generated/
make figures      # Figures 2-4 into paper/figures/
```

The studies' own verifiers (`make verify` in the original project, `provenance/verify_studies.py`), the replay of
each study's saved-output analysis (`analysis/replay_study_analyses.py`) and two post hoc analyses
(`analysis/posthoc_intervals.py`, `analysis/posthoc_query_feed.py`) read the raw run outputs, which hold about 18 GB
(full-vocabulary logits, cached signals, caches, residual states and traces) and are not included. Their saved
results are: the verifier passes recorded in each study's `analysis/` folder, `analysis/results/replay-check.json`
(all 30 saved analysis files reproduced exactly), `posthoc-intervals.json`, `query-feed.json` and
`query-feed-development.json`. They can be rerun after the runs are regenerated.

## Model reruns

Nothing in the manuscript needs the model to run again. To repeat a study, follow its `REPRODUCIBILITY.md`: copy
its `code/`, `protocol/`, `data/` and `provenance/` into a new sibling folder under `studies/`, with empty `runs/` and
`analysis/` folders, and run the listed stages in order. The runs used GPT-2 Medium in float32 with eager
attention on one 6 GB NVIDIA GPU; they need the CUDA build of torch, about 4 GiB of free GPU
memory and at least 3 GiB of free host memory. Studies 2 and 3 run every example in a fixed batch of eight, which
made their replays bitwise identical; Study 1's single-example replay differed by up to 0.0012 in residual
coordinates (tolerance 0.0005), while its replay with the original batches was bitwise identical.

```sh
make models                       # GPT-2 Medium at the pinned revision, every file checked (about 1.5 GB)
```

The study code takes the checkpoint from `MODELS_DIR` (default `models/`) and serialises GPU jobs with a lock
file named by `GPU_LOCK` (default `gpu.lock` in this folder). Study 1's `code/prepare_confirmation.py` also
excluded every prompt and name of an earlier project from the confirmation set; that project is not included, so
repeating that step needs `EARLIER_PROJECT_DIR` set to its folder.

## Tables and figures

| Manuscript | Source |
|---|---|
| All numbers in the text | macros in `paper/generated/numbers.tex`, written by `paper/make_numbers.py` |
| Table 1 (Study 3) | `paper/generated/table-study3.tex` |
| Table 3 (order contrast by wording) | `paper/generated/table-wording.tex` |
| Table 4 (freeze records) | `paper/generated/table-freezes.tex` |
| Table 5 (gates) | `paper/generated/table-gates.tex` |
| Table 6 (development and confirmation) | `paper/generated/table-devconf.tex` |
| Table 7 (Study 3 factorial) | `paper/generated/table-factorial.tex` |
| Table 8 (Study 3 screen) | `paper/generated/table-screen.tex` |
| Table 2 (templates) | written in the manuscript's source |
| Figures 2, 3 and 4 | `paper/make_figures.py` → `paper/figures/study1.pdf`, `study2.pdf`, `study3.pdf` (plotted values in `figure-data.json`) |
| Figure 1 | a diagram drawn in the manuscript's source |

The values come from each study's frozen analysis (`code/analyze_*.py` and the study summaries in `analysis/`) and
from the post hoc analyses in `analysis/`, labelled post hoc in the manuscript.

## What is not included

- The raw run outputs (about 18 GB) and six large saved arrays of Study 1 (per-pair responses and prediction
  errors of the development, holdout and confirmation banks; 262 MB); the study summaries derived from them are
  included.
- The model weights (download them; see `models/README.md`).
- The studies' written reports and figures, logs, the manuscript and its source, two working notes of the
  preparation stage, the earlier project's documentation and code that the studies recorded by hash as
  references, and the original project's provenance history. Frozen files not shipped are listed with their
  SHA-256 in `RELEASE-CHANGES.md`.

## Changes from the original project

Files are copied byte for byte, except those listed in `RELEASE-CHANGES.md` with the original's SHA-256: records
had personal paths, the names of other project folders and the names of software tools in working records replaced by placeholders such as `<project>`, `<earlier-project>` and `[redacted]` (the assistants used are named under "Use of AI assistants" below); the studies' model, lock and reference paths now point inside the
release (`MODELS_DIR`, `GPU_LOCK`, `reference/`); and the earlier project's folder for the prompt exclusion comes
from `EARLIER_PROJECT_DIR`.

## Licence

MIT (`LICENSE`), covering the code and the studies' records and saved outputs. GPT-2 Medium is distributed under its own licence (MIT).

## Study codes

The records and code name the studies by the codes they had while the work was done. Each code that occurs here,
and where that study is released:

| Code | Released in |
|---|---|
| AI-01 | `interp-answer-repair` |
| AI-07 | `interp-skill-updates` |
| AI-08 | this repository |

## Use of AI assistants

As the preprint states:

> Language-model assistants were used throughout this work, under the author's
> direction, to help design the studies, to write and run the code for the experiments and analyses, to draft and
> revise the text and to check the references against their online records. All reported results come from the
> saved study outputs, and the numbers in the text were generated from those outputs or checked against them.

In the working records shipped here, the names of software tools were replaced by placeholders such as
`[redacted]`.

## Citation

Please cite the preprint (the link will be added here when it is posted) and, for the code and results, this
repository; [CITATION.cff](CITATION.cff) has the details.

```bibtex
@software{interp_position_cue_2026,
  author  = {Jeevash, Richard},
  title   = {{A fixed internal signal acts as a relative-position cue: code and results}},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/Jeevash23/interp-position-cue}
}
```
