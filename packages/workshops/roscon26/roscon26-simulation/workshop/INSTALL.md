# Installation Guide

## Required platform

| Component | Requirement |
|---|---|
| Host | Ubuntu 24.04 with **ROCm 7.13** installed from AMD's apt packages (`amdrocm-*7.13`) |
| Hardware | AMD Strix Halo mini-PC (Ryzen AI MAX+ 395, GPU target `gfx1151`) |
| Python | 3.13 in a virtualenv created by `uv` (the wheel supports 3.12 and 3.13) |
| PyTorch | `2.11.0+rocm7.13.0` from `https://repo.amd.com/rocm/whl/gfx1151/` |
| Genesis | `genesis-world==1.4.0` with its compiler `quadrants==1.3.0` (installed through the gslab wheel) |
| gslab | Bundled `wheels/gslab-0.1.2-py3-none-any.whl`, installed as `gslab[rocm]` |


## 1. Create the Python environment

```bash
/opt/rocm/llvm/bin/ld.lld --version     # must report LLD 19 or newer
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.13 .venv
source .venv/bin/activate

uv pip install --index-url https://repo.amd.com/rocm/whl/gfx1151/ \
    "torch==2.11.0+rocm7.13.0"

uv pip install \
    --index-url https://pypi.org/simple \
    --extra-index-url https://repo.amd.com/rocm/whl/gfx1151/ \
    --index-strategy unsafe-best-match \
    "gslab @ file://$PWD/wheels/gslab-0.1.2-py3-none-any.whl" \
    jupyterlab ipykernel ipywidgets plotly tqdm

./link_rocm_tree.sh # link the ROCm tree for Genesis
```

The `link_rocm_tree.sh` script: 

1. finds the pip ROCm SDK inside `.venv` (`_rocm_sdk_core`, `_rocm_sdk_libraries_gfx1151`);
2. adds the unversioned `.so` aliases (`libamdhip64.so` next to `libamdhip64.so.7`, and so on);
3. creates `.venv/rocm` with `lib/`, `lib-gfx1151/`, `bin/` and `share/` linked to the pip SDK;
4. looks for an `ld.lld` of LLVM 19 or newer, preferring `/opt/rocm/llvm/bin/ld.lld`, then
   `lld-20`/`lld-19` from apt.llvm.org, and links it as `.venv/rocm/llvm/bin/ld.lld`.
   Set `LLD=/path/to/ld.lld` to choose one explicitly;
5. writes `.venv/rocm/env.sh`, which exports the variables below;
6. verifies that the linker runs and that `libamdhip64.so` loads by bare name.


## 2. Register the jupyter kernel

```bash
source .venv/bin/activate && source .venv/rocm/env.sh
python -m ipykernel install --user \
    --name gslab-roscon \
    --display-name "gslab ROSCon (ROCm 7.13)" \
    --env ROCM_PATH "$PWD/.venv/rocm" \
    --env LD_LIBRARY_PATH "$PWD/.venv/rocm/lib:$PWD/.venv/rocm/lib-gfx1151" \
    --env QD_OFFLINE_CACHE_MAX_SIZE_OF_FILES 2147483647 \
    --env PYTHONDONTWRITEBYTECODE 1
```

`env.sh` also raises the quadrants on-disk kernel-cache limit above its default so compiled
kernels persist between runs.

## 3. Launch JupyterLab

```bash
source .venv/bin/activate && source .venv/rocm/env.sh
jupyter lab --notebook-dir .
```
