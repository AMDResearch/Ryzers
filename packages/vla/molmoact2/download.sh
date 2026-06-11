#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Pre-fetch the MolmoAct2 weights + DROID dataset into the persistent HF cache so
# the demos don't download on first use. Optional — every demo also auto-downloads
# what it needs. Set HF_TOKEN for gated repos.
#
#   HF_TOKEN=hf_xxx ryzers run /ryzers/download.sh
set -euo pipefail

if [ -n "${HF_TOKEN:-}" ]; then
  hf auth login --token "$HF_TOKEN" >/dev/null 2>&1 || true
fi

echo "==> MolmoAct2-DROID (model, ~22 GB) — smoke + DROID demos"
hf download allenai/MolmoAct2-DROID
echo "==> MolmoAct2-DROID-Dataset — DROID open-loop demo"
hf download allenai/MolmoAct2-DROID-Dataset --repo-type dataset
echo "==> MolmoAct2-Think-LIBERO (model) — LIBERO closed-loop demo"
hf download allenai/MolmoAct2-Think-LIBERO
echo "PASS: assets cached under ${HF_HOME:-/root/.cache/huggingface}"
