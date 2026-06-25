#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Pre-fetch ALL MolmoAct2 demo weights into the persistent HF cache (both models). For a
# single demo, use the matching download_<demo>.sh to pull only what it needs. Optional:
# every demo also downloads what it needs on first run. Set HF_TOKEN for gated repos.
#
#   HF_TOKEN=hf_xxx ryzers run /ryzers/download.sh
#
# Note: the DROID open-loop demo fetches only the few dataset files for the chosen episode
# at run time, so the full DROID-Dataset is intentionally NOT pre-fetched here.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HF_COMMON:-$HERE/_hf_common.sh}"

hf_prefetch allenai/MolmoAct2-DROID         # smoke + DROID open-loop
hf_prefetch allenai/MolmoAct2-Think-LIBERO  # LIBERO closed-loop + interactive demos
echo "PASS: assets cached under ${HF_HOME:-/root/.cache/huggingface}"
