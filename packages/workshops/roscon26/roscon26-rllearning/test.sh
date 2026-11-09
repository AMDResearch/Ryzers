#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set -e

RL_VENV="${RL_VENV:-/opt/rl-venv}"
NOTEBOOKS=/ryzers/notebooks

echo "Testing ROSCon26 RL Learning workshop environment..."

echo "Python version: $(${RL_VENV}/bin/python --version)"

# JupyterLab arrives from the composed `jupyterlab` ryzer. The RL stack itself
# does not need it, so a standalone `ryzers build roscon26-rllearning` warns
# rather than fails.
echo "Checking JupyterLab..."
if command -v jupyter-lab &> /dev/null; then
    echo "JupyterLab version: $(jupyter-lab --version)"
    jupyter kernelspec list 2>/dev/null | grep -q roscon26-rl \
        && echo "Kernel registered: roscon26-rl" \
        || echo "Warning: roscon26-rl kernelspec not visible to this jupyter"
else
    echo "Warning: jupyter-lab not found - build as 'ryzers build jupyterlab roscon26-rllearning' to run the notebook"
fi

# ---------------------------------------------------------------------------
# RL stack. These are hard failures: without them the workshop cannot run.
# ---------------------------------------------------------------------------
echo "Checking RL stack..."
${RL_VENV}/bin/python - <<'PY'
import os
from importlib.metadata import version

os.environ["MUJOCO_GL"] = "disable"

import jax

backend = jax.default_backend()
devices = jax.devices()
print(f"JAX backend: {backend}")
print(f"JAX devices: {devices}")

for package in ("jax", "jaxlib", "mujoco", "mujoco-mjx", "brax", "playground"):
    print(f"  {package}: {version(package)}")

import brax  # noqa: F401
import mujoco  # noqa: F401
from mujoco_playground import registry

env = registry.load("PandaPickCube", config_overrides={"impl": "jax"})
print(f"PandaPickCube: obs={env.observation_size} act={env.action_size} dt={env.dt}")

if backend == "rocm" or any("rocm" in str(d).lower() for d in devices):
    print("ROCm device active")
else:
    print(f"Warning: no ROCm device, JAX fell back to '{backend}'. "
          "Rollouts still run, just on CPU.")
PY

# ---------------------------------------------------------------------------
# Checkpoints the notebook rolls out in section 4.
# ---------------------------------------------------------------------------
echo "Checking checkpoints..."
for checkpoint in \
    assets/PandaPickCube-20260807-131132/checkpoints/000008192000 \
    assets/PandaPickCube-20260817-150103/checkpoints/000006553600 \
    assets/PandaPickCube-20260817-150103/checkpoints/000045875200
do
    if [[ -f "${NOTEBOOKS}/${checkpoint}/ppo_network_config.json" ]]; then
        echo "  OK: ${checkpoint}"
    else
        echo "FAIL: missing checkpoint ${NOTEBOOKS}/${checkpoint}"
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Render backend. Section 4 renders in a subprocess, so probe it the same way
# the notebook does rather than trusting that the libraries are installed.
# ---------------------------------------------------------------------------
echo "Checking render backend..."
cd "${NOTEBOOKS}"
${RL_VENV}/bin/python - <<'PY'
import sys
from pathlib import Path

root = Path("/ryzers/notebooks")
sys.path.insert(0, str(root / "assets"))
from headless_gl import probe_render_subprocess

can_render, report, _ = probe_render_subprocess(root)
if not can_render:
    print("FAIL: no working render backend. Backends tried:\n")
    print(report)
    raise SystemExit(1)
print(f"Render backend: {report}")
PY

echo "Checking notebooks..."
echo "Notebooks found: $(find "${NOTEBOOKS}" -name '*.ipynb' | wc -l)"

echo "SUCCESS: ROSCon26 RL Learning workshop test passed"
echo "Note: for the full rollout + video, run: /ryzers/demo_roscon26_rllearning.sh"
echo "Note: for JupyterLab, run: jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
exit 0
