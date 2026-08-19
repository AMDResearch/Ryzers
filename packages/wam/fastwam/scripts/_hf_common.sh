#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Shared HF pre-fetch helper for download_*.sh: source it, then call
# `hf_prefetch <repo> [files...] [--local-dir DIR]`.
#
# FastWAM pins huggingface-hub==0.29.2 (CLI `huggingface-cli`); Xet off + retries for
# reliable anonymous downloads. Set HF_HUB_ENABLE_HF_TRANSFER=1 for parallel speed.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"

if [ -n "${HF_TOKEN:-}" ]; then
  huggingface-cli login --token "$HF_TOKEN" >/dev/null 2>&1 || true
fi

hf_prefetch() {
  local tries="${DL_RETRIES:-5}" n=1
  echo "==> prefetch: $* (DISABLE_XET=$HF_HUB_DISABLE_XET HF_TRANSFER=$HF_HUB_ENABLE_HF_TRANSFER timeout=${HF_HUB_DOWNLOAD_TIMEOUT}s)"
  while true; do
    if huggingface-cli download "$@"; then echo "    cached: $1"; return 0; fi
    if [ "$n" -ge "$tries" ]; then
      echo "ERROR: 'huggingface-cli download $*' failed after $tries attempts" >&2
      return 1
    fi
    echo "    (attempt $n/$tries failed; retrying in 5s, resuming from cache...)" >&2
    n=$((n + 1)); sleep 5
  done
}
