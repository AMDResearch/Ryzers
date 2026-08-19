#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Real-time interactive RoboTwin demo (FastWAM policy over HTTP/MJPEG); arms HOLD while the
# model plans so planner latency is visible. Requires the simulation/robotwin base:
#   ryzers build robotwin fastwam
#   ryzers run /ryzers/demos/demo_interactive_robotwin_rt.sh
# View at http://localhost:PORT (ssh -L PORT:localhost:PORT <host>).
set -euo pipefail
if [ ! -d /opt/sim/sim_robotwin ]; then
  echo "ERROR: simulation/robotwin base not found (no /opt/sim/sim_robotwin)." >&2
  echo "       Build the chain:  ryzers build robotwin fastwam" >&2
  exit 1
fi
export TASK="${TASK:-click_bell}"
export TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
export SEED="${SEED:-100000}"
export PORT="${PORT:-8083}"
export OUT_DIR="${OUT_DIR:-/outputs}"
export CKPT="${CKPT:-/models/fastwam_release/robotwin_uncond_3cam_384.pt}"
export DATASET_STATS="${DATASET_STATS:-/models/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json}"
export POLICY_FACTORY="fastwam_robotwin_policy:build_policy"
export PYTHONPATH="/opt/fastwam-adapters:/repos/fastwam/src:/repos/fastwam:/repos/fastwam/experiments/robotwin:/opt/sim:/opt/RoboTwin:${PYTHONPATH:-}"

bash /ryzers/scripts/setup_robotwin.sh
exec python -m sim_robotwin.interactive_server_rt
