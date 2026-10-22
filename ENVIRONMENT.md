# Environment

The studies ran in one Python environment, recorded exactly in `uv.lock` (180 locked packages, including the
release itself as a virtual project). `pyproject.toml` declares it the same way the original project did: the
`experiment` extra is the full research environment, and the `dev` group adds pytest.

| Package | Version |
|---|---|
| Python | 3.12.14 (`uv` installs it if it is missing) |
| torch | 2.13.0 (CUDA 13 build from PyPI) |
| transformers | 5.16.1 |
| numpy | 2.5.2 |
| scipy | 1.18.1 |
| matplotlib | 3.11.1 |
| safetensors | 0.8.0 |
| huggingface-hub | 1.29.0 |

The study code imports torch, transformers, numpy and scipy; the number and figure scripts need only numpy,
scipy and matplotlib. The other packages in the lock (JupyterLab, datasets, TransformerLens and their dependencies)
were part of the environment and are kept so that the lock is the one the runs used.

## A fresh install

```sh
uv sync --frozen --extra experiment --group dev        # what make setup runs
```

On Linux x86-64 this downloads 176 packages, about 3.0 GB of wheels. The largest are torch (527 MB) and the
NVIDIA CUDA libraries it depends on (about 2.0 GB together: cuBLAS 423 MB, cuDNN 366 MB, cuFFT 214 MB, NCCL 206
MB, cuSOLVER 201 MB, cuSPARSELt 170 MB, cuSPARSE 146 MB and smaller ones), then triton (198 MB). The sizes come
from `uv.lock`; installing creates `.venv/` in this folder.

## A smaller CPU-only environment

The quick check and the number and figure scripts need only NumPy, SciPy and Matplotlib:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv numpy==2.5.2 scipy==1.18.1 matplotlib==3.11.1
```

For model reruns add torch 2.13.0 (with CUDA) and transformers 5.16.1, or use `make setup`.

This is not the locked environment; for model reruns that are meant to reproduce the recorded numbers to the
protocols' tolerances, use `make setup`.
