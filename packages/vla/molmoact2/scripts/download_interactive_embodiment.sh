#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Pre-fetch ONLY the weights demo_interactive_embodiment.sh needs (allenai/MolmoAct2-Think-LIBERO) into the
# persistent HF cache. Optional: demo_interactive_embodiment.sh also downloads them on first run.
# Set HF_TOKEN for gated repos.
#
#   HF_TOKEN=hf_xxx ryzers run /ryzers/download_interactive_embodiment.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HF_COMMON:-$HERE/_hf_common.sh}"

hf_prefetch allenai/MolmoAct2-Think-LIBERO
echo "PASS: assets cached under ${HF_HOME:-/root/.cache/huggingface}"
