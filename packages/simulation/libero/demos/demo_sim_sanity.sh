#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Model-free sanity rollout: rolls the built-in RandomPolicy through one scene on the
# iGPU (headless EGL) and saves an MP4. Exercises the ROCm/EGL render + MuJoCo + encode path.
#   ryzers run /ryzers/demos/demo_sim_sanity.sh
#   SUITE=libero_goal TASK_ID=2 STEPS=120 ryzers run /ryzers/demos/demo_sim_sanity.sh
set -euo pipefail
export SUITE="${SUITE:-libero_object}"
export TASK_ID="${TASK_ID:-0}"
export SEED="${SEED:-1000}"
export STEPS="${STEPS:-80}"
export OUT_DIR="${OUT_DIR:-/sim_outputs}"
exec python -m sim_libero.sanity
