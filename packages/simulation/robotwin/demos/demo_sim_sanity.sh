#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Model-free sanity rollout: fetch assets, run RandomPolicy on one task, save a 4-view MP4.
# Proves the ROCm/Vulkan SAPIEN render + mplib execution + encode path with no model.
#   ryzers run /ryzers/demos/demo_sim_sanity.sh
#   TASK=lift_pot MAX_STEPS=80 ryzers run /ryzers/demos/demo_sim_sanity.sh
set -euo pipefail
export TASK="${TASK:-click_bell}"
export TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
export SEED="${SEED:-100000}"
export MAX_STEPS="${MAX_STEPS:-60}"
export OUT_DIR="${OUT_DIR:-/sim_outputs}"

bash /ryzers/scripts/setup_robotwin.sh
exec python -m sim_robotwin.sanity
