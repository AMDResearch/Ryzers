#!/usr/bin/env bash

# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set -euo pipefail

echo "Testing WAN 2.2 Ryzer install..."

python3 - <<'PY'
import importlib
import pathlib

import imageio
import numpy as np
import torch
import torch.nn.functional as F

print("torch", torch.__version__)
print("hip", getattr(torch.version, "hip", None))
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("ROCm PyTorch did not expose a cuda-compatible device")
print("device", torch.cuda.get_device_name(0))

q = torch.randn((1, 4, 64, 64), device="cuda", dtype=torch.bfloat16)
k = torch.randn((1, 4, 64, 64), device="cuda", dtype=torch.bfloat16)
v = torch.randn((1, 4, 64, 64), device="cuda", dtype=torch.bfloat16)
out = F.scaled_dot_product_attention(q, k, v)
if not torch.isfinite(out).all():
    raise SystemExit("SDPA smoke produced non-finite output")

flash_attn = importlib.import_module("flash_attn")
print("flash_attn", getattr(flash_attn, "__version__", "unknown"))

from wan.configs import SUPPORTED_SIZES, WAN_CONFIGS

assert "ti2v-5B" in WAN_CONFIGS
assert "1280*704" in SUPPORTED_SIZES["ti2v-5B"]
print("wan_ti2v_config", WAN_CONFIGS["ti2v-5B"].__name__)

import subprocess
import sys

download_help_result = subprocess.run(
    [sys.executable, "/ryzers/download_wan22.py", "--help"],
    check=True,
    capture_output=True,
    text=True,
)
if "TI2V-5B" not in download_help_result.stdout:
    raise SystemExit("download entry point help did not advertise TI2V-5B")

help_result = subprocess.run(
    [sys.executable, "/ryzers/generate_ti2v_strix.py", "--help"],
    check=True,
    capture_output=True,
    text=True,
)
if "1280x704" not in help_result.stdout:
    raise SystemExit("generation entry point help did not advertise native TI2V size")

frames = np.zeros((2, 32, 32, 3), dtype=np.uint8)
out_path = pathlib.Path("/tmp/wan22_ryzer_test.mp4")
imageio.mimsave(out_path, list(frames), fps=1)
if not out_path.exists() or out_path.stat().st_size == 0:
    raise SystemExit("mp4 write failed")

print("WAN 2.2 Ryzer smoke PASS")
PY

