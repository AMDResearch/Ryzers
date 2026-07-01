# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Build the single-rollout LIBERO artifacts from a lerobot-eval run dir:
  * copies the rollout video to $OUT/libero_<suite>_<taskid>.mp4
  * plots the executed 7-DoF action trajectory (from the AMD-PATCH *_actions.npy)
    to $OUT/libero_<suite>_<taskid>_actions.png

Closed-loop sim eval => no reference trajectory, so we plot the executed actions
only (workspace rule 2.b: fully generative task, no GT overlay).

Usage: libero_action_plot.py <run_dir> <out_dir> <suite> <task_id>
"""
import json
import os
import shutil
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DIMS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def main() -> int:
    run_dir, out_dir, suite, task_id = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{suite}_{task_id}"
    vdir = os.path.join(run_dir, "videos", tag)

    success = None
    info_path = os.path.join(run_dir, "eval_info.json")
    if os.path.exists(info_path):
        info = json.load(open(info_path))
        for t in info.get("per_task", []):
            if t.get("task_group") == suite and str(t.get("task_id")) == str(task_id):
                s = t.get("metrics", {}).get("successes", [])
                success = bool(s[0]) if s else None

    mp4_src = os.path.join(vdir, "eval_episode_0.mp4")
    if os.path.exists(mp4_src):
        shutil.copyfile(mp4_src, os.path.join(out_dir, f"libero_{tag}.mp4"))
        print(f"video            : {os.path.join(out_dir, f'libero_{tag}.mp4')}")
    else:
        print(f"WARN: no rollout video at {mp4_src}", file=sys.stderr)

    npy = os.path.join(vdir, "eval_episode_0_actions.npy")
    if not os.path.exists(npy):
        print(f"WARN: no actions .npy at {npy}", file=sys.stderr)
        return 0
    a = np.load(npy)  # (T, 7)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), squeeze=False)
    for d in range(min(a.shape[1], 7)):
        ax = axes[d // 4][d % 4]
        ax.plot(a[:, d], color="tab:purple", lw=1.3)
        ax.set_title(DIMS[d], fontsize=10)
        ax.tick_params(labelsize=7)
    axes[1][3].axis("off")
    status = {True: "SUCCESS", False: "FAIL", None: "n/a"}[success]
    color = {True: "#2d7d34", False: "#b52b27", None: "#444"}[success]
    fig.suptitle(f"MolmoAct2-Think-LIBERO closed-loop (ROCm gfx1151)\n"
                 f"suite={suite}  task_id={task_id}  ->  {status}  ({a.shape[0]} steps)",
                 fontsize=12, color=color)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_png = os.path.join(out_dir, f"libero_{tag}_actions.png")
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"action plot      : {out_png}")
    print(f"result           : suite={suite} task_id={task_id} -> {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
