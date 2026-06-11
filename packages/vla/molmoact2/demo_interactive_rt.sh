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
#   THINK=1 ryzers run /ryzers/demo_interactive_rt.sh   # full depth reasoning (slower)
#
# Only if you run this on a REMOTE/headless box, forward the port first:
#   ssh -L 8081:localhost:8081 <host>
#
# FAST MODE by default: depth reasoning OFF (THINK=0) + num_steps=4. The overnight
# closed-loop sweep showed this keeps 100% success on libero_object while running
# ~2.9x faster than the full depth-reasoning path (which also adds a ~20s first-plan
# stall) — so the real-time view actually flows. Set THINK=1 for the full "Think"
# spatial-reasoning path (slower; better on harder spatial tasks).
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
