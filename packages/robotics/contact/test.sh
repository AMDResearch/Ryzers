#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set -euo pipefail

dataset_path="${CONTACT_DATASET:-/datasets/contact/barbed_flat}"
output_root="${CONTACT_OUTPUT_DIR:-/outputs/contact}"

if [ ! -f "${dataset_path}/.zgroup" ]; then
    echo "CONTACT dataset not found at ${dataset_path}" >&2
    echo "Place the dataset in workspace/contact/data/barbed_flat." >&2
    exit 1
fi

python - <<'PY'
import torch

if torch.version.hip is None:
    raise RuntimeError(f"PyTorch is not a ROCm build: {torch.__version__}")
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot access the ROCm GPU")

device = torch.device("cuda")
properties = torch.cuda.get_device_properties(device)
print(
    f"PyTorch {torch.__version__} | HIP {torch.version.hip} | "
    f"{properties.name} ({properties.gcnArchName})"
)
PY

cd /workspace/CONTACT
python scripts/smoke_train_amd.py

run_dir="${output_root}/smoke-$(date -u +%Y%m%d-%H%M%S)"
python train.py \
    --config-name=train_diffusion_workspace_disassembly.yaml \
    task=visff_disassembly \
    "dataset_path=${dataset_path}" \
    training.enable_rollout=false \
    training.device=auto \
    training.seed=42 \
    training.resume=false \
    training.num_epochs=1 \
    training.max_train_steps=1 \
    training.max_val_steps=1 \
    training.val_every=1 \
    training.sample_every=1 \
    training.checkpoint_every=10000 \
    checkpoint.save_last_ckpt=false \
    logging.mode=disabled \
    dataloader.batch_size=1 \
    dataloader.num_workers=0 \
    val_dataloader.batch_size=1 \
    val_dataloader.num_workers=0 \
    policy.num_inference_steps=10 \
    "hydra.run.dir=${run_dir}"

python - "${run_dir}/logs.json.txt" <<'PY'
import json
import math
import pathlib
import sys

log_path = pathlib.Path(sys.argv[1])
records = [
    json.loads(line)
    for line in log_path.read_text().splitlines()
    if line.strip()
]
metrics = records[-1]
required = ("train_loss", "val_loss", "train_action_mse_error")
for key in required:
    value = metrics.get(key)
    if value is None or not math.isfinite(value):
        raise RuntimeError(f"Invalid {key}: {value}")

print(
    "CONTACT dataset training passed | "
    + " | ".join(f"{key}={metrics[key]:.8f}" for key in required)
)
PY
