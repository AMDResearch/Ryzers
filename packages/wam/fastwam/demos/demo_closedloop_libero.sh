#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Closed-loop LIBERO rollouts in MuJoCo (headless EGL) driven by FastWAM. Writes
# per-task results + rollout videos to /outputs/closedloop/<TAG> and an aggregate
# success_summary.json. Set VISUALIZE_FUTURE=true for the slow path that also
# renders the model's imagined future video alongside each rollout.
#   ryzers run /ryzers/demos/demo_closedloop_libero.sh
#   NUM_TASKS=1 NUM_TRIALS=3 VISUALIZE_FUTURE=true ryzers run /ryzers/demos/demo_closedloop_libero.sh
set -uo pipefail
if [ ! -d /opt/LIBERO ]; then
  echo "ERROR: simulation/libero base not found (no /opt/LIBERO)." >&2
  echo "       Build the chain:  ryzers build libero fastwam" >&2
  exit 1
fi
SUITE="${SUITE:-libero_object}"
NUM_TASKS="${NUM_TASKS:-10}"
NUM_TRIALS="${NUM_TRIALS:-20}"
VISUALIZE_FUTURE="${VISUALIZE_FUTURE:-false}"
TAG="${TAG:-cl_${SUITE}}"
REL=/models/fastwam_release
OUT="${OUT_DIR:-/outputs}/closedloop/$TAG"
export PYTHONPATH="/repos/fastwam/src:/repos/fastwam:/repos/fastwam/experiments/libero:/opt/LIBERO:${PYTHONPATH:-}"

CKPT=$REL/libero_uncond_2cam224.pt
[ -f "$CKPT" ] || { echo "missing $CKPT -> run scripts/download_checkpoints.sh libero" >&2; exit 1; }

cd /tmp
for t in $(seq 0 $((NUM_TASKS-1))); do
  echo "########## $SUITE task $t ##########"
  python /repos/fastwam/experiments/libero/eval_libero_single.py \
    ckpt=$CKPT gpu_id=0 mixed_precision=bf16 \
    EVALUATION.task_suite_name=$SUITE EVALUATION.task_id=$t EVALUATION.num_trials=$NUM_TRIALS \
    EVALUATION.visualize_future_video=$VISUALIZE_FUTURE EVALUATION.output_dir=$OUT \
    EVALUATION.dataset_stats_path=$REL/libero_uncond_2cam224_dataset_stats.json \
    2>&1 | grep -E "Task [0-9]+ completed|Saved rollout|Traceback|Error executing|success="
done

python3 - "$OUT/$SUITE" <<'PY'
import sys, json, glob, os
d = sys.argv[1]
files = sorted(glob.glob(os.path.join(d, "gpu*_task*_results.json")))
tot_s = tot_n = 0; rows = []
for f in files:
    r = json.load(open(f))
    s = int(r.get("successes", 0)); n = int(r.get("total_episodes", 0))
    tot_s += s; tot_n += n; rows.append((r.get("task_id"), s, n, r.get("task_description")))
for tid, s, n, desc in sorted(rows, key=lambda x: (x[0] if x[0] is not None else 0)):
    print(f"  task {tid}: {s}/{n} ({100.0*s/n if n else 0:.1f}%) {str(desc)[:50]}")
overall = (100.0*tot_s/tot_n) if tot_n else 0.0
print(f"OVERALL: {tot_s}/{tot_n} ({overall:.1f}%)")
json.dump({"suite": os.path.basename(d), "overall_successes": tot_s, "overall_episodes": tot_n,
           "overall_success_rate_pct": round(overall, 2),
           "per_task": [{"task_id": tid, "successes": s, "episodes": n,
                         "success_rate_pct": round(100.0*s/n, 2) if n else 0.0, "task": desc}
                        for tid, s, n, desc in rows]},
          open(os.path.join(d, "success_summary.json"), "w"), indent=2)
print("wrote", os.path.join(d, "success_summary.json"))
PY
