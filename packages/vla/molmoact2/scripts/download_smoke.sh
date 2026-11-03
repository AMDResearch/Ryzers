#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Pre-fetch ONLY the weights demo_smoke.sh needs (allenai/MolmoAct2-DROID) into the
# persistent HF cache. Optional: demo_smoke.sh also downloads them on first run.
# Set HF_TOKEN for gated repos.
#
#   HF_TOKEN=hf_xxx ryzers run /ryzers/download_smoke.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HF_COMMON:-$HERE/_hf_common.sh}"

hf_prefetch allenai/MolmoAct2-DROID
echo "PASS: assets cached under ${HF_HOME:-/root/.cache/huggingface}"
