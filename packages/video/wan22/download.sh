#!/usr/bin/env bash

# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set -euo pipefail

CKPT_DIR="${CKPT_DIR:-/models/Wan2.2-TI2V-5B}"
HF_REPO_ID="${HF_REPO_ID:-Wan-AI/Wan2.2-TI2V-5B}"

python3 /ryzers/download_wan22.py \
    --repo_id "$HF_REPO_ID" \
    --ckpt_dir "$CKPT_DIR"
