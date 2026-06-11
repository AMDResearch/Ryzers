#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Interactive demo: stream a live LIBERO sim to your browser and type instructions
# for the MolmoAct2-LIBERO policy in real time. The sim sits IDLE on a scene until you
# send an instruction; it then resets the env + policy and runs that one task, streaming
# the live camera and saving a debug video. Local inference only — runs entirely on this
# machine, no external server. The container uses host networking (see config.yaml), so
# the browser server is reachable on localhost directly.
#
#   ryzers run /ryzers/demo_interactive.sh        # then open http://localhost:8080
#   SUITE=libero_object TASK_ID=3 PORT=8080 ryzers run /ryzers/demo_interactive.sh
#
# Only if you run this on a REMOTE/headless box, forward the port to your laptop first:
#   ssh -L 8080:localhost:8080 <host>
set -euo pipefail

export SUITE="${SUITE:-libero_object}"
export TASK_ID="${TASK_ID:-3}"
export SEED="${SEED:-1000}"
# Optional fast path (mirrors the RT demo): FAST=1 turns off depth reasoning and
# uses 4 flow-matching steps so frames render faster. THINK / NUM_STEPS still win
# if set explicitly.
export FAST="${FAST:-0}"
if [ "$FAST" = "1" ]; then
  THINK="${THINK:-0}"
  NUM_STEPS="${NUM_STEPS:-4}"
fi
export THINK="${THINK:-1}"
export NUM_STEPS="${NUM_STEPS:-}"
export CKPT="${CKPT:-allenai/MolmoAct2-Think-LIBERO}"
export PORT="${PORT:-8080}"
export PYTHONUNBUFFERED=1

PY=/opt/libero-venv/bin/python
[ -x "$PY" ] || PY=python
SERVER="${SERVER:-/ryzers/interactive_server.py}"
[ -f "$SERVER" ] || SERVER="$(dirname "$0")/interactive_server.py"

echo "Interactive demo | suite=$SUITE task_id=$TASK_ID seed=$SEED think=$THINK fast=$FAST num_steps=${NUM_STEPS:-default} port=$PORT"
echo "Open http://localhost:$PORT in your browser (remote box: ssh -L $PORT:localhost:$PORT <host>)"
exec "$PY" "$SERVER"
