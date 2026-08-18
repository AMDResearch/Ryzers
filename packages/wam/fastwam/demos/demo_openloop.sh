#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Open-loop replay: feed ground-truth observations from released LeRobot episodes
# and overlay predicted vs GT action chunks. Writes per-episode overlays, aggregate
# MAE graphs and summary.json to /outputs/openloop/openloop_<TAG>.
#   DATASET=libero   ryzers run /ryzers/demos/demo_openloop.sh
#   DATASET=robotwin NUM_EPISODES=100 ryzers run /ryzers/demos/demo_openloop.sh
set -euo pipefail
DATASET="${DATASET:-libero}"
REL=/models/fastwam_release
export FASTWAM_REPO=/repos/fastwam
export PYTHONPATH="/repos/fastwam/src:/repos/fastwam:${PYTHONPATH:-}"
export NUM_EPISODES="${NUM_EPISODES:-100}"
export NUM_STEPS="${NUM_STEPS:-20}"
export OUT_DIR="${OUT_DIR:-/outputs}/openloop"

case "$DATASET" in
  libero)
    export CONFIG_NAME=sim_libero TAG="${TAG:-libero_object}"
    export CKPT=$REL/libero_uncond_2cam224.pt
    export DATASET_STATS=$REL/libero_uncond_2cam224_dataset_stats.json
    export DATASET_DIR=/models/data/libero_object_no_noops_lerobot ;;
  robotwin)
    export CONFIG_NAME=sim_robotwin TAG="${TAG:-robotwin}"
    export CKPT=$REL/robotwin_uncond_3cam_384.pt
    export DATASET_STATS=$REL/robotwin_uncond_3cam_384_dataset_stats.json
    export DATASET_DIR=/models/data/robotwin2.0 ;;
  *) echo "DATASET must be libero|robotwin" >&2; exit 2 ;;
esac

[ -f "$CKPT" ] || { echo "missing $CKPT -> run scripts/download_checkpoints.sh $DATASET" >&2; exit 1; }
[ -d "$DATASET_DIR" ] || { echo "missing $DATASET_DIR -> run scripts/download_datasets.sh $DATASET" >&2; exit 1; }

exec python /ryzers/scripts/openloop_replay.py
