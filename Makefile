# A fixed internal signal acts as a relative-position cue: code release.
# `make help` lists the targets. PYTHON defaults to the environment that `make setup` (uv sync) creates.
PYTHON ?= .venv/bin/python
PY := $(if $(findstring /,$(PYTHON)),$(abspath $(PYTHON)),$(PYTHON))
export PYTHONDONTWRITEBYTECODE := 1
export OPENBLAS_NUM_THREADS ?= 2
export OMP_NUM_THREADS ?= 2
.NOTPARALLEL:

.PHONY: help setup python-check models unpack check-freezes quick-check numbers figures

help:
	@echo "make setup        create .venv with the pinned environment (uv sync; downloads packages, see ENVIRONMENT.md)"
	@echo "make quick-check  CPU only, under a minute, no model and no network: freezes, every manuscript number and"
	@echo "                  generated table, and every plotted value, recomputed from the saved outputs"
	@echo "make numbers      rewrite paper/generated/ from the saved outputs"
	@echo "make figures      draw Figures 2-4 into paper/figures/"
	@echo "make models       download GPT-2 Medium (about 1.5 GB) for model reruns"

setup:
	uv sync --frozen --extra experiment --group dev

python-check:
	@$(PY) -c "import numpy, scipy, matplotlib" 2>/dev/null || { echo "No usable Python at $(PYTHON): run make setup, or set PYTHON=..."; exit 1; }

models: python-check
	$(PY) tools/get_models.py gpt2-medium

unpack: python-check
	$(PY) tools/unpack.py

check-freezes: unpack
	$(PY) tools/check_freezes.py

quick-check: unpack
	CUDA_VISIBLE_DEVICES= $(PY) tools/check_freezes.py
	CUDA_VISIBLE_DEVICES= $(PY) -B paper/make_numbers.py --check
	CUDA_VISIBLE_DEVICES= $(PY) -B paper/make_figures.py --check

numbers: unpack
	$(PY) -B paper/make_numbers.py

figures: unpack
	$(PY) -B paper/make_figures.py
