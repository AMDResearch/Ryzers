#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Real-time interactive RoboTwin demo served over HTTP/MJPEG. Execution is decoupled from
# planning so the browser SEES the planner latency: the arms HOLD (THINKING) while the
# policy plans, then resume when the action buffer refills. Defaults to the built-in
# RandomPolicy; a policy image chains on top and sets POLICY_FACTORY for a real model.
#   ryzers run /ryzers/demos/demo_interactive_rt.sh
#   POLICY_FACTORY=fastwam_robotwin_policy:build_policy ryzers run /ryzers/demos/demo_interactive_rt.sh
set -euo pipefail
export TASK="${TASK:-click_bell}"
export TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
export SEED="${SEED:-100000}"
export PORT="${PORT:-8083}"
export OUT_DIR="${OUT_DIR:-/sim_outputs}"

bash /ryzers/scripts/setup_robotwin.sh
exec python -m sim_robotwin.interactive_server_rt
