#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Interactive demo: stream a live LIBERO sim to your browser and type instructions
# for the MolmoAct2-LIBERO policy in real time. The sim sits idle on a scene until you
# send an instruction; it then resets the env + policy and runs that one task, streaming
# the live camera and saving a debug video. Local inference only, running on this
# machine, no external server. The container uses host networking (see config.yaml), so
# the browser server is reachable on localhost directly.
#
#   ryzers run /ryzers/demo_interactive.sh        # Panda, then open http://localhost:8080
#   EMBODIMENT=ur5e ryzers run /ryzers/demo_interactive.sh
#   EMBODIMENT=xarm6 ryzers run /ryzers/demo_interactive.sh
#
# Only if you run this on a REMOTE/headless box, forward the port to your laptop first:
#   ssh -L 8080:localhost:8080 <host>
set -euo pipefail

export SUITE="${SUITE:-libero_object}"
export TASK_ID="${TASK_ID:-3}"
export SEED="${SEED:-1000}"
export THINK="${THINK:-0}"
export NUM_STEPS="${NUM_STEPS:-4}"
export CKPT="${CKPT:-allenai/MolmoAct2-Think-LIBERO}"
export PORT="${PORT:-8080}"
export PYTHONUNBUFFERED=1

EMBODIMENT="${EMBODIMENT:-}"
case "${EMBODIMENT,,}" in
  ur5e)
    export EMBODIMENT_GRIPPER="${EMBODIMENT_GRIPPER:-Robotiq85Gripper}"
    export EMBODIMENT_UR5E_ROBOTIQ_INIT_BLEND="${EMBODIMENT_UR5E_ROBOTIQ_INIT_BLEND:-0.5}"
    export EMBODIMENT_UR5E_ROBOTIQ_EEF_OFFSET="${EMBODIMENT_UR5E_ROBOTIQ_EEF_OFFSET:-0}"
    export EMBODIMENT_UR5E_ROBOTIQ_CAMERA_OFFSET="${EMBODIMENT_UR5E_ROBOTIQ_CAMERA_OFFSET:-0}"
    ;;
  xarm6)
    export EMBODIMENT_GRIPPER="${EMBODIMENT_GRIPPER:-XArmGripper}"
    export EMBODIMENT_EXECUTOR="${EMBODIMENT_EXECUTOR:-absolute}"
    export EMBODIMENT_SERVO_STEPS="${EMBODIMENT_SERVO_STEPS:-2}"
    export EMBODIMENT_XARM6_CAMERA_MATCH="${EMBODIMENT_XARM6_CAMERA_MATCH:-1}"
    ;;
esac

PY=/opt/libero-venv/bin/python
[ -x "$PY" ] || PY=python
SERVER="${SERVER:-/ryzers/interactive_server.py}"
[ -f "$SERVER" ] || SERVER="$(dirname "$0")/interactive_server.py"

echo "Interactive demo | arm=${EMBODIMENT:-panda} suite=$SUITE task_id=$TASK_ID seed=$SEED think=$THINK num_steps=$NUM_STEPS port=$PORT"
echo "Open http://localhost:$PORT in your browser (remote box: ssh -L $PORT:localhost:$PORT <host>)"
if [ -n "$EMBODIMENT" ] && [ "${EMBODIMENT,,}" != "panda" ]; then
  EMB_DIR="$(dirname "$SERVER")"
  [ -f "$EMB_DIR/embodiment.py" ] || EMB_DIR="$(dirname "$0")"
  exec "$PY" -c "import sys, runpy; sys.path.insert(0, '$EMB_DIR'); import embodiment; r=embodiment.apply_from_env(); print('[embodiment] LIBERO arm ->', r, flush=True); runpy.run_path('$SERVER', run_name='__main__')"
fi
exec "$PY" "$SERVER"
