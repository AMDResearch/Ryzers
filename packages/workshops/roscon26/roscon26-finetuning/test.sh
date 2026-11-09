#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Environment check for the ROSCon26 Fine-tuning workshop. Imports only: it
# never downloads a model and never runs a rollout, so it finishes in well under
# a minute. The full model path is exercised by demo_roscon26_finetuning.sh.

set -e

TRAIN_VENV="${TRAIN_VENV:-/opt/train-venv}"
FASTWAM_VENV="${FASTWAM_VENV:-/opt/fastwam-venv}"
NOTEBOOKS=/ryzers/notebooks

echo "Testing ROSCon26 Fine-tuning workshop environment..."
echo "train-venv   python: $(${TRAIN_VENV}/bin/python --version)"
echo "fastwam-venv python: $(${FASTWAM_VENV}/bin/python --version)"

# ---------------------------------------------------------------------------
# JupyterLab arrives from the composed `jupyterlab` ryzer. The workshop stacks
# themselves do not need it, so a standalone build warns rather than fails.
# ---------------------------------------------------------------------------
echo "Checking JupyterLab..."
if command -v jupyter-lab &> /dev/null; then
    echo "JupyterLab version: $(jupyter-lab --version)"
    jupyter kernelspec list 2>/dev/null | grep -q roscon26-finetuning \
        && echo "Kernel registered: roscon26-finetuning" \
        || echo "Warning: roscon26-finetuning kernelspec not visible to this jupyter"
else
    echo "Warning: jupyter-lab not found - build as 'ryzers build jupyterlab roscon26-finetuning' to run the notebooks"
fi

# ---------------------------------------------------------------------------
# ROCm torch. A missing /dev/kfd only warns: the image is still correct, it just
# cannot run the course. A non-ROCm torch is a hard failure.
# ---------------------------------------------------------------------------
echo "Checking ROCm torch..."
${TRAIN_VENV}/bin/python - <<'PY'
import sys

import torch

print(f"torch            : {torch.__version__}  hip={torch.version.hip}")
if not torch.version.hip:
    print("FAIL: torch is not a ROCm build.", file=sys.stderr)
    raise SystemExit(1)

if not torch.cuda.is_available():
    print("Warning: no ROCm device visible. Check --device=/dev/kfd, /dev/dri "
          "and --group-add render. Every notebook needs one.")
    raise SystemExit(0)

print(f"device[0]        : {torch.cuda.get_device_name(0)}")
a = torch.randn(512, 512, device="cuda")
b = torch.randn(512, 512, device="cuda")
print(f"matmul ok        : sum={(a @ b).sum().item():.3f}")
PY

# ---------------------------------------------------------------------------
# MolmoAct2 training stack (notebooks 1 and 2). Hard failures: without these the
# fine-tune and the interactive sim cannot run.
# ---------------------------------------------------------------------------
echo "Checking MolmoAct2 training stack..."
${TRAIN_VENV}/bin/python - <<'PY'
import os
import subprocess
import sys
from pathlib import Path

import accelerate
import einops          # noqa: F401
import fastapi         # noqa: F401
import huggingface_hub
import json_numpy      # noqa: F401
import lerobot
import mujoco
import robosuite
import safetensors     # noqa: F401
import sentencepiece   # noqa: F401
import torch
import transformers
from packaging.version import Version

from lerobot.envs.libero import _get_suite
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy  # noqa: F401
from libero.libero import get_assets_path, get_libero_path

# The DROID policy loader notebook 1 imports, patched at build time to accept an
# already-resolved local snapshot path as well as a Hub repo id.
sys.path.insert(0, os.environ["DROID_SERVER_DIR"])
from host_server_droid import NORM_TAG, Policy  # noqa: E402

assert torch.version.hip, f"torch is not a ROCm build: {torch.__version__}"
assert lerobot.__version__ == "0.5.2", lerobot.__version__
assert Version("5.4") <= Version(transformers.__version__) < Version("5.6")
assert Policy.__name__ == "Policy" and NORM_TAG

# The exact config notebook 1 fine-tunes with. Constructing it (no weights) is
# enough to catch a policy-config rename in the lerobot fork.
cfg = MolmoAct2Config(
    checkpoint_path="allenai/MolmoAct2-DROID",
    train_mode_vlm="lora",
    action_mode="both",
    chunk_size=10,
    n_action_steps=10,
    setup_type="single franka robotic arm in libero",
    control_mode="delta end-effector pose",
    image_keys=[
        "observation.images.image",
        "observation.images.wrist_image",
    ],
    model_dtype="bfloat16",
    num_flow_timesteps=8,
    gradient_checkpointing=True,
    freeze_embedding=True,
    normalize_gripper=False,
    enable_knowledge_insulation=False,
    push_to_hub=False,
)
assert cfg.train_mode_vlm == "lora"
assert cfg.action_mode == "both"

# LIBERO simulator metadata, baked at build under LIBERO_CONFIG_PATH.
for key in ("benchmark_root", "bddl_files", "init_states"):
    assert Path(get_libero_path(key)).exists(), (key, get_libero_path(key))
assert Path(get_assets_path()).exists(), get_assets_path()
suite = _get_suite("libero_object")
assert len(suite.tasks) == 10

# Every training flag notebook 1 passes must exist on the CLI.
help_result = subprocess.run(
    [sys.executable, "-m", "lerobot.scripts.lerobot_train", "--policy.type=molmoact2", "--help"],
    check=True,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)
required_flags = {
    "--policy.train_mode_vlm",
    "--policy.setup_type",
    "--policy.control_mode",
    "--policy.image_keys",
    "--policy.model_dtype",
    "--policy.num_flow_timesteps",
    "--policy.freeze_embedding",
    "--policy.normalize_gripper",
    "--policy.enable_knowledge_insulation",
}
missing = sorted(flag for flag in required_flags if flag not in help_result.stdout)
assert not missing, f"MolmoAct2 training CLI is missing: {missing}"

print(f"python           : {sys.executable}")
print(f"lerobot          : {lerobot.__version__}")
print(f"transformers     : {transformers.__version__}")
print(f"accelerate       : {accelerate.__version__}")
print(f"huggingface_hub  : {huggingface_hub.__version__}")
print(f"robosuite/mujoco : {robosuite.__version__} / {mujoco.__version__}")
print(f"libero tasks     : {len(suite.tasks)}")
print("PASS: MolmoAct2 inference, training CLI, and LIBERO metadata OK")
PY

# Notebook 1 step 5 calls this by absolute path.
if [[ -x "${TRAIN_VENV}/bin/lerobot-eval" ]]; then
    echo "lerobot-eval     : ${TRAIN_VENV}/bin/lerobot-eval"
else
    echo "FAIL: ${TRAIN_VENV}/bin/lerobot-eval is missing"
    exit 1
fi

# ---------------------------------------------------------------------------
# FastWAM world-action model + its own LIBERO sim stack (notebook 3). Runs in
# the isolated venv on its own LIBERO config path, so it never touches the
# train-venv checked above.
# ---------------------------------------------------------------------------
echo "Checking FastWAM world-action stack..."
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH_FASTWAM:-/opt/libero-config-fastwam}" \
${FASTWAM_VENV}/bin/python - <<'PY'
import os
import sys

import numpy as np
import torch
import mujoco
import robosuite  # noqa: F401

from libero.libero import benchmark  # noqa: F401
from libero.libero.envs import OffScreenRenderEnv  # noqa: F401

import sim_libero  # noqa: F401
from sim_libero.libero_env import SUITES, get_benchmark_dict, get_max_steps
from sim_libero.policy import Policy, load_policy  # noqa: F401

import fastwam  # noqa: F401
from fastwam.runtime import create_fastwam  # noqa: F401
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json  # noqa: F401
import experiments.libero.eval_libero_single as E  # noqa: F401

# The runtime policy factory the interactive server selects via POLICY_FACTORY.
sys.path.insert(0, "/opt/fastwam-adapters")
import fastwam_libero_policy  # noqa: E402
assert hasattr(fastwam_libero_policy, "build_policy"), "fastwam_libero_policy.build_policy missing"

assert torch.version.hip, f"torch is not a ROCm build: {torch.__version__}"
assert np.__version__.startswith("1.26"), f"expected numpy 1.26.x in fastwam venv, got {np.__version__}"

benchmark_dict = get_benchmark_dict()
for suite in ("libero_object", "libero_goal", "libero_spatial", "libero_10"):
    assert suite in benchmark_dict, f"suite {suite} missing from LIBERO benchmark dict"
    get_max_steps(suite)

rel = os.environ.get("FASTWAM_RELEASE_DIR", "/root/.cache/fastwam/fastwam_release")
ckpt = os.path.join(rel, "libero_uncond_2cam224.pt")
stats = os.path.join(rel, "libero_uncond_2cam224_dataset_stats.json")
diffsynth = os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", "/root/.cache/fastwam/diffsynth")

print(f"python           : {sys.executable}")
print(f"torch            : {torch.__version__}  hip={torch.version.hip}")
print(f"numpy            : {np.__version__}")
print(f"mujoco/robosuite : {mujoco.__version__} / {robosuite.__version__}")
print(f"libero suites    : {', '.join(SUITES)}")
# These are runtime downloads, not image content: absent on a fresh cache.
print(f"cached ckpt      : {ckpt} -> {'PRESENT' if os.path.exists(ckpt) else 'absent (downloads on first use)'}")
print(f"cached stats     : {stats} -> {'PRESENT' if os.path.exists(stats) else 'absent (downloads on first use)'}")
print(f"Wan2.2 base      : {diffsynth} -> {'PRESENT' if os.path.isdir(diffsynth) else 'absent (downloads on first use)'}")
print("PASS: FastWAM world-action + LIBERO sim imports OK")
PY

# ---------------------------------------------------------------------------
# Course content.
# ---------------------------------------------------------------------------
echo "Checking course content..."
for f in \
    0_overview.ipynb \
    1_finetune_molmoact2_libero.ipynb \
    2_interactive_sim_molmoact2_libero.ipynb \
    3_inference_fastwam_libero.ipynb \
    scripts/ft_helpers.py \
    scripts/fast_to_device.py \
    scripts/interactive_server_ft.py \
    scripts/interactive_server_fastwam.py \
    scripts/fastwam_videogen.py \
    scripts/fastwam_smoke.py
do
    if [[ -f "${NOTEBOOKS}/${f}" ]]; then
        echo "  OK: ${f}"
    else
        echo "FAIL: missing ${NOTEBOOKS}/${f}"
        exit 1
    fi
done

for d in "${OUT_DIR:-/ryzers/outputs}" "${CHECKPOINTS_DIR:-/ryzers/checkpoints}" "${HF_HOME:-/root/.cache/huggingface}"; do
    mkdir -p "${d}"
    [[ -w "${d}" ]] || { echo "FAIL: ${d} is not writable"; exit 1; }
    echo "  writable: ${d}"
done

echo "SUCCESS: ROSCon26 Fine-tuning workshop test passed"
echo "Note: for a GPU model smoke test (downloads ~25 GB on first run), run: /ryzers/demo_roscon26_finetuning.sh"
echo "Note: for JupyterLab, run: jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
exit 0
