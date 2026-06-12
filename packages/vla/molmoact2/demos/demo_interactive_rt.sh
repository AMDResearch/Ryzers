#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Real-time interactive demo: the LIBERO sim runs at wall-clock speed while the
# policy plans asynchronously, so the arm holds its pose while thinking and moves
# when the next action chunk lands. Defaults to the fast path (depth reasoning
# off, 4 denoising steps). Set THINK=1 for full depth reasoning, RT_HORIZON_MULT
# to extend the sim horizon on long tasks.
#
#   ryzers run /ryzers/demo_interactive_rt.sh     # open http://localhost:8081
#   SUITE=libero_object TASK_ID=3 ryzers run /ryzers/demo_interactive_rt.sh
#   THINK=1 ryzers run /ryzers/demo_interactive_rt.sh
set -euo pipefail

export SUITE="${SUITE:-libero_object}"
export TASK_ID="${TASK_ID:-3}"
export SEED="${SEED:-1000}"
export THINK="${THINK:-0}"
export NUM_STEPS="${NUM_STEPS:-4}"
export CKPT="${CKPT:-allenai/MolmoAct2-Think-LIBERO}"
export PORT="${PORT:-8081}"
export VIEW_RES="${VIEW_RES:-720}"
export RT_HZ="${RT_HZ:-20}"
export RT_LOOKAHEAD="${RT_LOOKAHEAD:-0}"
export RT_HORIZON_MULT="${RT_HORIZON_MULT:-1}"
export PYTHONUNBUFFERED=1

PY=/opt/libero-venv/bin/python
[ -x "$PY" ] || PY=python
SERVER="${SERVER:-/ryzers/interactive_server_rt.py}"
[ -f "$SERVER" ] || SERVER="$(dirname "$0")/interactive_server_rt.py"

echo "Real-time demo | suite=$SUITE task_id=$TASK_ID seed=$SEED think=$THINK num_steps=$NUM_STEPS port=$PORT hz=$RT_HZ"
echo "Open http://localhost:$PORT in your browser (remote box: ssh -L $PORT:localhost:$PORT <host>)"
exec "$PY" "$SERVER"
