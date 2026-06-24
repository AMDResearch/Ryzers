#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Pre-fetch the MolmoAct2 weights + DROID dataset into the persistent HF cache so
# the demos don't download on first use. Optional because every demo also downloads
# what it needs. Set HF_TOKEN for gated repos.
#
#   HF_TOKEN=hf_xxx ryzers run /ryzers/download.sh
#
# Robustness: these repos are Xet-stored; the plain-HTTP fallback serves from the Xet
# bridge CDN, which rate-limits CONCURRENT anonymous requests (403 'no permits
# available'). hf_transfer's 3 parallel streams trip that at the first uncached shard,
# so we default to a single stream (HF_HUB_ENABLE_HF_TRANSFER=0) + Xet client off, and
# retry so a dropped connection resumes from cache. Set HF_HUB_ENABLE_HF_TRANSFER=1
# to go parallel where permits/auth allow it.
set -euo pipefail

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"

if [ -n "${HF_TOKEN:-}" ]; then
  hf auth login --token "$HF_TOKEN" >/dev/null 2>&1 || true
fi

# hf download resumes partial blobs, so repeated attempts keep making progress.
dl() {
  local tries="${DL_RETRIES:-5}" n=1
  while true; do
    if hf download "$@"; then return 0; fi
    if [ "$n" -ge "$tries" ]; then
      echo "ERROR: 'hf download $*' failed after $tries attempts" >&2
      return 1
    fi
    echo "  (attempt $n/$tries failed; retrying in 5s, resuming from cache...)" >&2
    n=$((n + 1)); sleep 5
  done
}

echo "==> HF backend: DISABLE_XET=$HF_HUB_DISABLE_XET HF_TRANSFER=$HF_HUB_ENABLE_HF_TRANSFER timeout=${HF_HUB_DOWNLOAD_TIMEOUT}s"
echo "==> MolmoAct2-DROID (model, ~22 GB): smoke + DROID demos"
dl allenai/MolmoAct2-DROID
echo "==> MolmoAct2-DROID-Dataset: DROID open-loop demo"
dl allenai/MolmoAct2-DROID-Dataset --repo-type dataset
echo "==> MolmoAct2-Think-LIBERO (model): LIBERO closed-loop demo"
dl allenai/MolmoAct2-Think-LIBERO
echo "PASS: assets cached under ${HF_HOME:-/root/.cache/huggingface}"
