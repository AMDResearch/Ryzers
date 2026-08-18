#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Model-free sanity rollout for the RoboTwin simulator base. Fetches the RoboTwin sim
# assets on first run, then loads the built-in RandomPolicy (no weights), sets up one task
# scene on the iGPU (offscreen Vulkan RT), rolls it via mplib TOPP, and saves an MP4 of the
# live 4-view (head|observer over left|right wrist) to /sim_outputs. Proves the ROCm/Vulkan
# SAPIEN render + mplib execution + 4-view compose + video encode path work with no model
# (the same runtime path the closed-loop eval uses; the curobo expert pre-check is off).
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
