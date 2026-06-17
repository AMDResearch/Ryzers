#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Capability 3: LIBERO closed-loop rollout. Pick a random scene (suite + task),
# run the MolmoAct2-Think-LIBERO policy in the LIBERO/robosuite MuJoCo simulator
# (EGL headless on the iGPU), and write the rollout video + executed 7-DoF action
# plot under /outputs.
#
#   ryzers run /ryzers/demo_libero.sh                            # random scene
#   SUITE=libero_object TASK_ID=3 ryzers run /ryzers/demo_libero.sh
#   THINK=0 ryzers run /ryzers/demo_libero.sh                    # base (no depth reasoning)
set -euo pipefail

OUT_DIR="${OUT_DIR:-/outputs}"
CKPT="${CKPT:-allenai/MolmoAct2-Think-LIBERO}"
THINK="${THINK:-1}"
SEED_ARG="${SEED:-$RANDOM}"
SUITES=(libero_10 libero_goal libero_object libero_spatial)
SUITE="${SUITE:-${SUITES[$((RANDOM % ${#SUITES[@]}))]}}"
TASK_ID="${TASK_ID:-$((RANDOM % 10))}"

bool() { [ "$1" = "1" ] && echo True || echo False; }
DEPTH=$(bool "$THINK"); ADAPT=$(bool "$THINK")

# Optional flow-matching denoising steps (lower = slightly faster; blank = model default).
NUM_STEPS="${NUM_STEPS:-}"
NSTEP_ARG=()
[ -n "$NUM_STEPS" ] && NSTEP_ARG=(--policy.num_steps="$NUM_STEPS")

RUN="$OUT_DIR/_libero_${SUITE}_${TASK_ID}_seed${SEED_ARG}"
mkdir -p "$RUN"
echo "LIBERO closed-loop demo | suite=$SUITE task_id=$TASK_ID seed=$SEED_ARG think=$THINK num_steps=${NUM_STEPS:-default} ckpt=$CKPT"

LEROBOT_EVAL=/opt/libero-venv/bin/lerobot-eval
[ -x "$LEROBOT_EVAL" ] || LEROBOT_EVAL=lerobot-eval

"$LEROBOT_EVAL" \
  --policy.type=molmoact2 \
  --policy.checkpoint_path="$CKPT" \
  --policy.inference_action_mode=continuous \
  --policy.enable_depth_reasoning="$DEPTH" \
  --policy.enable_adaptive_depth="$ADAPT" \
  --policy.enable_cuda_graph=False \
  --policy.norm_tag=libero \
  --policy.device=cuda \
  "${NSTEP_ARG[@]}" \
  --env.type=libero \
  --env.task="$SUITE" \
  --env.task_ids="[$TASK_ID]" \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --seed="$SEED_ARG" \
  --output_dir="$RUN/run"

python /ryzers/libero_action_plot.py "$RUN/run" "$OUT_DIR" "$SUITE" "$TASK_ID"
echo "PASS: artifacts in $OUT_DIR"
