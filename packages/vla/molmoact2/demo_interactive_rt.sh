#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Real-time (_RT) interactive demo: the LIBERO sim runs at wall-clock speed while the
# MolmoAct2 policy plans asynchronously, so you can SEE the planner latency — the robot
# holds its pose while "thinking", then moves when the next action chunk lands. This is
# the real-time sibling of demo_interactive.sh (which fast-replays a clean chunk rollout).
# Local inference only; host networking (config.yaml) makes it reachable on localhost.
#
#   ryzers run /ryzers/demo_interactive_rt.sh           # then open http://localhost:8081
#   SUITE=libero_object TASK_ID=3 PORT=8081 ryzers run /ryzers/demo_interactive_rt.sh
#   RT_HZ=20 RT_LOOKAHEAD=0 ryzers run /ryzers/demo_interactive_rt.sh
#
# Only if you run this on a REMOTE/headless box, forward the port first:
#   ssh -L 8081:localhost:8081 <host>
set -euo pipefail

export SUITE="${SUITE:-libero_object}"
export TASK_ID="${TASK_ID:-3}"
export SEED="${SEED:-1000}"
export THINK="${THINK:-1}"
export CKPT="${CKPT:-allenai/MolmoAct2-Think-LIBERO}"
export PORT="${PORT:-8081}"
export VIEW_RES="${VIEW_RES:-720}"
export RT_HZ="${RT_HZ:-20}"
export RT_LOOKAHEAD="${RT_LOOKAHEAD:-0}"
export PYTHONUNBUFFERED=1

PY=/opt/libero-venv/bin/python
[ -x "$PY" ] || PY=python
SERVER="${SERVER:-/ryzers/interactive_server_rt.py}"
[ -f "$SERVER" ] || SERVER="$(dirname "$0")/interactive_server_rt.py"

echo "Real-time demo | suite=$SUITE task_id=$TASK_ID seed=$SEED think=$THINK port=$PORT hz=$RT_HZ"
echo "Open http://localhost:$PORT in your browser (remote box: ssh -L $PORT:localhost:$PORT <host>)"
exec "$PY" "$SERVER"
