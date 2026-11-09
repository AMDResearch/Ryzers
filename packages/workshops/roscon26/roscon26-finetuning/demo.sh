#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# End-to-end GPU checks for the ROSCon26 Fine-tuning workshop. Unlike test.sh
# these load real weights, so each mode DOWNLOADS a large public model on its
# first run and caches it under the host directories config.yaml mounts.
#
# Usage: demo_roscon26_finetuning.sh [molmoact2|fastwam|sim]
#
#   molmoact2  ~22 GB  load allenai/MolmoAct2-DROID, one flow-matching forward
#   fastwam    ~12 GB  load the released FastWAM LIBERO checkpoint + the Wan2.2
#                      T5/VAE components, one fast-route infer_action
#   sim        ~11 GB  serve allenai/MolmoAct2-LIBERO in the interactive LIBERO
#                      simulator at http://localhost:${RT_PORT:-8080}/
#
# Everything the notebooks do is also available interactively; this script is
# the headless equivalent for validating an image on real hardware.

set -e

TRAIN_VENV="${TRAIN_VENV:-/opt/train-venv}"
FASTWAM_VENV="${FASTWAM_VENV:-/opt/fastwam-venv}"
NOTEBOOKS=/ryzers/notebooks
MODE="${1:-molmoact2}"

cd "${NOTEBOOKS}"
mkdir -p "${OUT_DIR:-/ryzers/outputs}"

case "${MODE}" in

  molmoact2)
    echo "== MolmoAct2 DROID base: download (~22 GB on first run) + one forward =="
    "${TRAIN_VENV}/bin/python" - <<'PY'
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/ryzers/notebooks/scripts")
from ft_helpers import prefetch

# Same accelerated .to(device) path notebook 1 installs before loading the base.
from fast_to_device import install
install()

sys.path.insert(0, os.environ["DROID_SERVER_DIR"])
from host_server_droid import NORM_TAG, Policy

assert torch.cuda.is_available(), "no ROCm device visible (check /dev/kfd, /dev/dri)"
print(f"device[0]        : {torch.cuda.get_device_name(0)}")

# Same immutable Hub revision notebook 1 pins.
base = prefetch(
    os.environ.get("BASE_REPO", "allenai/MolmoAct2-DROID"),
    "model",
    revision=os.environ.get("BASE_REVISION", "d8c1abd8a27d8e859455bbe514df2bcc617db0fb"),
)

t0 = time.time()
policy = Policy(repo_id=base, device="cuda:0", dtype=torch.bfloat16)
params = sum(p.numel() for p in policy.model.parameters())
print(f"model loaded     : {time.time() - t0:.1f}s  params={params / 1e9:.2f}B  norm_tag={NORM_TAG}")

# One dummy DROID observation: 3 cameras (ext1, ext2, wrist) + an 8-DoF state.
images = [
    Image.fromarray(np.random.randint(0, 255, (180, 320, 3), dtype=np.uint8))
    for _ in range(3)
]
kwargs = dict(
    processor=policy.processor,
    images=images,
    task="pick up the object",
    state=np.zeros(8, np.float32),
    norm_tag=NORM_TAG,
    enable_depth_reasoning=False,
    num_steps=int(os.environ.get("NUM_STEPS", "10")),
    normalize_language=True,
    enable_cuda_graph=False,
)

t1 = time.time()
try:
    out = policy.model.predict_action(inference_action_mode="continuous", **kwargs)
except TypeError as exc:  # older checkpoints spell this `action_mode`
    if "action_mode" not in str(exc):
        raise
    out = policy.model.predict_action(action_mode="continuous", **kwargs)

raw = out.actions if hasattr(out, "actions") else out
if torch.is_tensor(raw):
    raw = raw.detach().to(dtype=torch.float32, device="cpu").numpy()
actions = np.asarray(raw, dtype=np.float32)
if actions.ndim == 3 and actions.shape[0] == 1:
    actions = actions[0]

print(f"predict_action   : {actions.shape} in {(time.time() - t1) * 1000:.0f} ms")
assert actions.ndim == 2 and actions.shape[-1] == 8 and np.isfinite(actions).all(), actions.shape
print("PASS: MolmoAct2 full-model ROCm smoke OK")
PY
    ;;

  fastwam)
    echo "== FastWAM: download the public release (~12 GB on first run) + one infer_action =="
    # The replay dataset is only needed by the video-imagination section of
    # notebook 3, so this skips its 1.35 GB archive.
    "${FASTWAM_VENV}/bin/python" - <<'PY'
import os
import sys

sys.path.insert(0, "/ryzers/notebooks/scripts")
from ft_helpers import prepare_fastwam_assets

prepare_fastwam_assets(
    os.environ["FASTWAM_RELEASE_DIR"],
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"],
    os.environ["FASTWAM_DATA_DIR"],
    include_dataset=False,
)
PY
    LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH_FASTWAM:-/opt/libero-config-fastwam}" \
      "${FASTWAM_VENV}/bin/python" scripts/fastwam_smoke.py
    ;;

  sim)
    RT_PORT="${RT_PORT:-8080}"
    echo "== Interactive LIBERO sim with the public MolmoAct2-LIBERO policy =="
    echo "   The container shares the host network, so once the policy has"
    echo "   loaded the UI is at http://localhost:${RT_PORT}/ (Ctrl-C to stop)."
    PORT="${RT_PORT}" \
    CKPT="${CKPT:-allenai/MolmoAct2-LIBERO}" \
    SUITE="${SUITE:-libero_object}" \
    TASK_ID="${TASK_ID:-3}" \
      exec "${TRAIN_VENV}/bin/python" scripts/interactive_server_ft.py
    ;;

  *)
    echo "Unknown mode '${MODE}'. Use one of: molmoact2, fastwam, sim"
    exit 2
    ;;
esac

echo "Demo complete: ${MODE}"
