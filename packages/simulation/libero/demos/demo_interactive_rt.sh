#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Real-time LIBERO demo: sim steps at wall-clock RT_HZ and the robot HOLDs while the
# policy plans, so latency is visible. Default RandomPolicy; set POLICY_FACTORY for a model.
# View at http://localhost:PORT (ssh -L PORT:localhost:PORT <host>).
#   ryzers run /ryzers/demos/demo_interactive_rt.sh
#   POLICY_FACTORY=fastwam_libero_policy:build_policy ryzers run /ryzers/demos/demo_interactive_rt.sh
set -euo pipefail
export SUITE="${SUITE:-libero_object}"
export TASK_ID="${TASK_ID:-0}"
export SEED="${SEED:-1000}"
export PORT="${PORT:-8081}"
export RT_HZ="${RT_HZ:-20}"
export OUT_DIR="${OUT_DIR:-/sim_outputs}"
exec python -m sim_libero.interactive_server_rt
