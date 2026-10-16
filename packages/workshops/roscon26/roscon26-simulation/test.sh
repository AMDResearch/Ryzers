#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set -e

GSLAB_VENV="${GSLAB_VENV:-/opt/gslab-venv}"
WORKSHOP=/ryzers/notebooks

echo "Testing ROSCon26 Simulation workshop environment..."

echo "Python version: $(${GSLAB_VENV}/bin/python --version)"

# JupyterLab arrives from the composed `jupyterlab` ryzer. The simulation stack
# itself does not need it, so a standalone build warns rather than fails.
echo "Checking JupyterLab..."
if command -v jupyter-lab &> /dev/null; then
    echo "JupyterLab version: $(jupyter-lab --version)"
    jupyter kernelspec list 2>/dev/null | grep -q gslab-roscon \
        && echo "Kernel registered: gslab-roscon" \
        || echo "Warning: gslab-roscon kernelspec not visible to this jupyter"
else
    echo "Warning: jupyter-lab not found - build as 'ryzers build jupyterlab roscon26-simulation' to run the notebooks"
fi

# ---------------------------------------------------------------------------
# The ROCm tree quadrants needs. Checked before importing Genesis, because a
# missing alias or an old linker shows up here as a clear message rather than
# as a kernel-compilation failure deep inside a notebook.
# ---------------------------------------------------------------------------
echo "Checking ROCm tree for Genesis..."
ROCM_TREE="${GSLAB_VENV}/rocm"
LIB_DIR="$(ls -d ${ROCM_TREE}/lib-* 2>/dev/null | head -1)"

if [[ ! -d "${ROCM_TREE}" ]]; then
    echo "FAIL: ${ROCM_TREE} missing - link_rocm_tree.sh did not run"
    exit 1
fi
if [[ -z "${LIB_DIR}" ]]; then
    echo "FAIL: no ${ROCM_TREE}/lib-<gfx> directory"
    exit 1
fi
echo "  rocm tree:  ${ROCM_TREE}"
echo "  libraries:  $(basename "${LIB_DIR}")"
echo "  ld.lld:     $(${ROCM_TREE}/llvm/bin/ld.lld --version | head -1)"

export ROCM_PATH="${ROCM_TREE}"
export LD_LIBRARY_PATH="${ROCM_TREE}/lib:${LIB_DIR}"
export QD_OFFLINE_CACHE_MAX_SIZE_OF_FILES=2147483647
export PYGLET_HEADLESS=1

${GSLAB_VENV}/bin/python -c \
  'import ctypes; ctypes.CDLL("libamdhip64.so"); print("  dlopen(libamdhip64.so): OK")'

# ---------------------------------------------------------------------------
# Simulation stack. Hard failures: the workshop cannot run without these.
# ---------------------------------------------------------------------------
echo "Checking simulation stack..."
${GSLAB_VENV}/bin/python - <<'PY'
import torch
assert torch.version.hip, f"torch is not a ROCm build: {torch.__version__}"
print(f"  torch: {torch.__version__} (hip {torch.version.hip})")

import genesis as gs
import gslab

print(f"  genesis: {gs.__version__}")
print(f"  gslab: {getattr(gslab, '__version__', 'installed')}")

from gslab.tasks.registry import list_tasks
tasks = list_tasks()
print(f"  gslab tasks: {len(tasks)}")

if torch.cuda.is_available():
    properties = torch.cuda.get_device_properties(0)
    print(f"  accelerator: {properties.name} ({properties.total_memory / 2**30:.1f} GiB)")
else:
    print("  Warning: no GPU visible to torch. Genesis needs one to compile "
          "and run its kernels, so the notebooks will not work here.")
PY

# ---------------------------------------------------------------------------
# Course layout. workshop_helpers.find_workshop_root() walks up from the
# notebook's cwd looking for a directory holding BOTH INSTALL.md and
# checkpoints/, so the nesting matters as much as the files.
# ---------------------------------------------------------------------------
echo "Checking workshop layout..."
for required in \
    INSTALL.md \
    checkpoints/g1_flat_baseline.pt \
    checkpoints/g1_stairs_easy_trained.pt \
    checkpoints/g1_ref.pt \
    notebooks/workshop_helpers.py \
    notebooks/workshop_training.py
do
    if [[ -e "${WORKSHOP}/${required}" ]]; then
        echo "  OK: ${required}"
    else
        echo "FAIL: missing ${WORKSHOP}/${required}"
        exit 1
    fi
done

cd "${WORKSHOP}/notebooks"
${GSLAB_VENV}/bin/python - <<'PY'
import sys
sys.path.insert(0, "/ryzers/notebooks/notebooks")
from workshop_helpers import find_workshop_root, workshop_paths

root = find_workshop_root()
paths = workshop_paths(root)
print(f"  workshop root resolves to: {root}")
for label in ("baseline", "supplied_trained", "reference"):
    path = getattr(paths, label)
    assert path.is_file(), f"missing {label}: {path}"
    print(f"  {label}: {path.relative_to(root)}")
PY

echo "Notebooks found: $(find "${WORKSHOP}" -name '*.ipynb' | wc -l)"

echo "SUCCESS: ROSCon26 Simulation workshop test passed"
echo "Note: for JupyterLab, run: jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
echo "      then open notebooks/00_workshop_intro.ipynb with the 'gslab ROSCon' kernel"
exit 0
