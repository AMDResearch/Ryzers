#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# End-to-end check: load a Brax PPO checkpoint, roll out one PandaPickCube
# episode, and render it to mp4. Slower than test.sh because the first JAX
# JIT compile of the policy and the env step can take up to a minute.
#
# Usage: demo_roscon26_rllearning.sh [early|mid|final] [output.mp4]

set -e

RL_VENV="${RL_VENV:-/opt/rl-venv}"
NOTEBOOKS=/ryzers/notebooks
STAGE="${1:-final}"
OUTPUT="${2:-${NOTEBOOKS}/outputs/panda_pick_cube_${STAGE}.mp4}"

case "${STAGE}" in
  early) CHECKPOINT=assets/PandaPickCube-20260807-131132/checkpoints/000008192000 ;;
  mid)   CHECKPOINT=assets/PandaPickCube-20260817-150103/checkpoints/000006553600 ;;
  final) CHECKPOINT=assets/PandaPickCube-20260817-150103/checkpoints/000045875200 ;;
  *)
    echo "Unknown stage '${STAGE}'. Use one of: early, mid, final"
    exit 2
    ;;
esac

cd "${NOTEBOOKS}"
mkdir -p "$(dirname "${OUTPUT}")"

echo "Rolling out '${STAGE}' checkpoint: ${CHECKPOINT}"
"${RL_VENV}/bin/python" scripts/render_checkpoint_once.py \
    "${CHECKPOINT}" \
    -o "${OUTPUT}"

echo "Demo complete: ${OUTPUT}"
