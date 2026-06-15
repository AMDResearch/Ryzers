#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Interactive demo: type instructions to the MolmoAct2-LIBERO policy and watch a
# live LIBERO sim in the browser. Defaults to the fast path (depth reasoning off,
# 4 denoising steps). Set THINK=1 for full depth reasoning.
#
#   ryzers run /ryzers/demo_interactive.sh        # open http://localhost:8080
#   SUITE=libero_object TASK_ID=3 ryzers run /ryzers/demo_interactive.sh
#   THINK=1 ryzers run /ryzers/demo_interactive.sh
#   EMBODIMENT=ur5e ryzers run /ryzers/demo_interactive.sh   # swap Panda -> UR5e
set -euo pipefail

export SUITE="${SUITE:-libero_object}"
export TASK_ID="${TASK_ID:-3}"
export SEED="${SEED:-1000}"
export THINK="${THINK:-0}"
export NUM_STEPS="${NUM_STEPS:-4}"
export CKPT="${CKPT:-allenai/MolmoAct2-Think-LIBERO}"
export PORT="${PORT:-8080}"
# Robot arm: unset/panda = stock LIBERO Franka Panda; ur5e = cross-embodiment swap.
export EMBODIMENT="${EMBODIMENT:-}"
export EMBODIMENT_GRIPPER="${EMBODIMENT_GRIPPER:-PandaGripper}"
export EMBODIMENT_ROT_DEG="${EMBODIMENT_ROT_DEG:-}"
export PYTHONUNBUFFERED=1

PY=/opt/libero-venv/bin/python
[ -x "$PY" ] || PY=python
SERVER="${SERVER:-/ryzers/interactive_server.py}"
[ -f "$SERVER" ] || SERVER="$(dirname "$0")/interactive_server.py"

echo "Interactive demo | arm=${EMBODIMENT:-panda} suite=$SUITE task_id=$TASK_ID seed=$SEED think=$THINK num_steps=$NUM_STEPS port=$PORT"
echo "Open http://localhost:$PORT in your browser (remote box: ssh -L $PORT:localhost:$PORT <host>)"
# Apply the optional embodiment swap (no-op when EMBODIMENT is unset/panda), then
# run the stock server unchanged so its make_env() builds the selected arm.
ARM_DIR="$(dirname "$SERVER")"
exec "$PY" -c "import sys, runpy; sys.path.insert(0, '$ARM_DIR'); import embodiment; print('[embodiment] LIBERO arm ->', embodiment.apply_from_env() or 'Panda (default)', flush=True); runpy.run_path('$SERVER', run_name='__main__')"
