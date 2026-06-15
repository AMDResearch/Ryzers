#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# LIBERO closed-loop rollout: run MolmoAct2-Think-LIBERO in the LIBERO MuJoCo
# simulator (EGL headless) and write the rollout video plus the executed action plot.
#
#   ryzers run /ryzers/demo_libero.sh                            # random scene
#   SUITE=libero_object TASK_ID=3 ryzers run /ryzers/demo_libero.sh
#   THINK=0 ryzers run /ryzers/demo_libero.sh                    # no depth reasoning
#   EMBODIMENT=ur5e ryzers run /ryzers/demo_libero.sh            # swap Panda -> UR5e
set -euo pipefail

OUT_DIR="${OUT_DIR:-/outputs}"
CKPT="${CKPT:-allenai/MolmoAct2-Think-LIBERO}"
THINK="${THINK:-1}"
SEED_ARG="${SEED:-$RANDOM}"
SUITES=(libero_10 libero_goal libero_object libero_spatial)
SUITE="${SUITE:-${SUITES[$((RANDOM % ${#SUITES[@]}))]}}"
TASK_ID="${TASK_ID:-$((RANDOM % 10))}"
# Robot arm: unset/panda = stock LIBERO Franka Panda; ur5e = cross-embodiment swap.
export EMBODIMENT="${EMBODIMENT:-}"
export EMBODIMENT_GRIPPER="${EMBODIMENT_GRIPPER:-PandaGripper}"
export EMBODIMENT_ROT_DEG="${EMBODIMENT_ROT_DEG:-}"

bool() { [ "$1" = "1" ] && echo True || echo False; }
DEPTH=$(bool "$THINK"); ADAPT=$(bool "$THINK")

# Optional flow-matching denoising steps (lower is a bit faster; blank uses default).
NUM_STEPS="${NUM_STEPS:-}"
NSTEP_ARG=()
[ -n "$NUM_STEPS" ] && NSTEP_ARG=(--policy.num_steps="$NUM_STEPS")

RUN="$OUT_DIR/_libero_${SUITE}_${TASK_ID}_seed${SEED_ARG}"
mkdir -p "$RUN"
echo "LIBERO closed-loop demo | suite=$SUITE task_id=$TASK_ID seed=$SEED_ARG think=$THINK num_steps=${NUM_STEPS:-default} ckpt=$CKPT"

LEROBOT_EVAL=/opt/libero-venv/bin/lerobot-eval
[ -x "$LEROBOT_EVAL" ] || LEROBOT_EVAL=lerobot-eval
PY=/opt/libero-venv/bin/python
[ -x "$PY" ] || PY=python
export ARM_DIR=/ryzers
[ -f "$ARM_DIR/embodiment.py" ] || ARM_DIR="$(dirname "$0")"

EVAL_ARGS=(
  --policy.type=molmoact2
  --policy.checkpoint_path="$CKPT"
  --policy.inference_action_mode=continuous
  --policy.enable_depth_reasoning="$DEPTH"
  --policy.enable_adaptive_depth="$ADAPT"
  --policy.enable_cuda_graph=False
  --policy.norm_tag=libero
  --policy.device=cuda
  "${NSTEP_ARG[@]}"
  --env.type=libero
  --env.task="$SUITE"
  --env.task_ids="[$TASK_ID]"
  --eval.batch_size=1
  --eval.n_episodes=1
  --seed="$SEED_ARG"
  --output_dir="$RUN/run"
)

# Apply the optional embodiment swap (no-op when EMBODIMENT is unset/panda), then
# run the stock lerobot-eval unchanged so its make_env() builds the selected arm.
"$PY" - "$LEROBOT_EVAL" "${EVAL_ARGS[@]}" <<'PYEOF'
import os, runpy, sys
sys.path.insert(0, os.environ.get("ARM_DIR", "/ryzers"))
import embodiment
print("[embodiment] LIBERO arm ->", embodiment.apply_from_env() or "Panda (default)", flush=True)
target = sys.argv.pop(1)  # lerobot-eval entry; remaining argv is its CLI
runpy.run_path(target, run_name="__main__")
PYEOF

python /ryzers/libero_action_plot.py "$RUN/run" "$OUT_DIR" "$SUITE" "$TASK_ID"
echo "PASS: artifacts in $OUT_DIR"
