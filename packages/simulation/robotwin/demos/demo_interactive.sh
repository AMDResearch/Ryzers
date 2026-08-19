#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Interactive RoboTwin demo (chunk-replay, HTTP/MJPEG). Default RandomPolicy; set POLICY_FACTORY
# to drive a real model. View at http://localhost:PORT (ssh -L PORT:localhost:PORT <host>).
#   ryzers run /ryzers/demos/demo_interactive.sh
#   POLICY_FACTORY=fastwam_robotwin_policy:build_policy ryzers run /ryzers/demos/demo_interactive.sh
set -euo pipefail
export TASK="${TASK:-click_bell}"
export TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
export SEED="${SEED:-100000}"
export PORT="${PORT:-8082}"
export OUT_DIR="${OUT_DIR:-/sim_outputs}"

bash /ryzers/scripts/setup_robotwin.sh
exec python -m sim_robotwin.interactive_server
