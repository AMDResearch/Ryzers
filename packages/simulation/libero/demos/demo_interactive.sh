#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Interactive LIBERO demo (chunk-replay) served over HTTP/MJPEG. Defaults to the
# built-in RandomPolicy; a policy image chains on top and sets POLICY_FACTORY to drive
# it with a real model. View at http://localhost:PORT (ssh -L PORT:localhost:PORT <host>).
#   ryzers run /ryzers/demos/demo_interactive.sh
#   POLICY_FACTORY=fastwam_libero_policy:build_policy ryzers run /ryzers/demos/demo_interactive.sh
set -euo pipefail
export SUITE="${SUITE:-libero_object}"
export TASK_ID="${TASK_ID:-0}"
export SEED="${SEED:-1000}"
export PORT="${PORT:-8080}"
export OUT_DIR="${OUT_DIR:-/sim_outputs}"
exec python -m sim_libero.interactive_server
