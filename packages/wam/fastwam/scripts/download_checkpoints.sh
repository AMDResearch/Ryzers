#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Fetch released FastWAM checkpoints + stats (yuanty/fastwam) into the model volume.
# The Wan2.2 base is fetched separately by the upstream loader on first run.
#   ryzers run /ryzers/scripts/download_checkpoints.sh [libero|robotwin|all]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HF_COMMON:-$HERE/_hf_common.sh}"

WHICH="${1:-all}"
DEST="${FASTWAM_RELEASE_DIR:-/models/fastwam_release}"
mkdir -p "$DEST"

files=()
case "$WHICH" in
  libero)   files=(libero_uncond_2cam224.pt libero_uncond_2cam224_dataset_stats.json) ;;
  robotwin) files=(robotwin_uncond_3cam_384.pt robotwin_uncond_3cam_384_dataset_stats.json) ;;
  all)      files=(libero_uncond_2cam224.pt libero_uncond_2cam224_dataset_stats.json
                   robotwin_uncond_3cam_384.pt robotwin_uncond_3cam_384_dataset_stats.json) ;;
  *) echo "usage: download_checkpoints.sh [libero|robotwin|all]" >&2; exit 2 ;;
esac

hf_prefetch yuanty/fastwam "${files[@]}" --local-dir "$DEST"
echo "PASS: FastWAM checkpoints ($WHICH) cached under $DEST"
ls -la "$DEST"
