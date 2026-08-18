#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Video imagination (joint path): predict future frames from the first observation
# and write side-by-side GT-vs-imagined clips + per-clip latency to
# /outputs/videogen/videogen_<TAG>.
#   DATASET=libero   ryzers run /ryzers/demos/demo_videogen.sh
#   DATASET=robotwin NUM_VIDEOS=10 ryzers run /ryzers/demos/demo_videogen.sh
set -euo pipefail
DATASET="${DATASET:-libero}"
REL=/models/fastwam_release
export FASTWAM_REPO=/repos/fastwam
export PYTHONPATH="/repos/fastwam/src:/repos/fastwam:${PYTHONPATH:-}"
export NUM_VIDEOS="${NUM_VIDEOS:-10}"
export NUM_STEPS="${NUM_STEPS:-20}"
export OUT_DIR="${OUT_DIR:-/outputs}/videogen"

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

exec python /ryzers/scripts/videogen_joint.py
