#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Pre-fetch ONLY the weights demo_droid.sh needs (allenai/MolmoAct2-DROID) into the
# persistent HF cache. The DROID open-loop demo fetches just the few dataset files for
# the chosen episode at run time, so the full DROID-Dataset is intentionally NOT pulled
# here. Set HF_TOKEN for gated repos.
#
#   HF_TOKEN=hf_xxx ryzers run /ryzers/download_droid.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HF_COMMON:-$HERE/_hf_common.sh}"

hf_prefetch allenai/MolmoAct2-DROID
echo "PASS: assets cached under ${HF_HOME:-/root/.cache/huggingface}"
