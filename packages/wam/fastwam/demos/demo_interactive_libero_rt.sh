#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Real-time LIBERO demo (FastWAM policy): sim steps at RT_HZ and the robot HOLDs while
# FastWAM plans, so planner latency is visible. Requires the simulation/libero base:
#   ryzers build libero fastwam
#   ryzers run /ryzers/demos/demo_interactive_libero_rt.sh
# View at http://localhost:PORT (ssh -L PORT:localhost:PORT <host>).
set -euo pipefail
if [ ! -d /opt/sim/sim_libero ]; then
  echo "ERROR: simulation/libero base not found (no /opt/sim/sim_libero)." >&2
  echo "       Build the chain:  ryzers build libero fastwam" >&2
  exit 1
fi
export SUITE="${SUITE:-libero_object}"
export TASK_ID="${TASK_ID:-0}"
export SEED="${SEED:-1000}"
export PORT="${PORT:-8081}"
export RT_HZ="${RT_HZ:-20}"
export OUT_DIR="${OUT_DIR:-/outputs}"
export CKPT="${CKPT:-/models/fastwam_release/libero_uncond_2cam224.pt}"
export DATASET_STATS="${DATASET_STATS:-/models/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
export POLICY_FACTORY="fastwam_libero_policy:build_policy"
export PYTHONPATH="/opt/fastwam-adapters:/repos/fastwam:/repos/fastwam/experiments/libero:/opt/sim:/opt/LIBERO:${PYTHONPATH:-}"

bash /ryzers/scripts/download_checkpoints.sh libero
exec python -m sim_libero.interactive_server_rt
