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
set -euo pipefail

export SUITE="${SUITE:-libero_object}"
export TASK_ID="${TASK_ID:-3}"
export SEED="${SEED:-1000}"
export THINK="${THINK:-0}"
export NUM_STEPS="${NUM_STEPS:-4}"
export CKPT="${CKPT:-allenai/MolmoAct2-Think-LIBERO}"
export PORT="${PORT:-8080}"
export PYTHONUNBUFFERED=1

PY=/opt/libero-venv/bin/python
[ -x "$PY" ] || PY=python
SERVER="${SERVER:-/ryzers/interactive_server.py}"
[ -f "$SERVER" ] || SERVER="$(dirname "$0")/interactive_server.py"

echo "Interactive demo | suite=$SUITE task_id=$TASK_ID seed=$SEED think=$THINK num_steps=$NUM_STEPS port=$PORT"
echo "Open http://localhost:$PORT in your browser (remote box: ssh -L $PORT:localhost:$PORT <host>)"
exec "$PY" "$SERVER"
