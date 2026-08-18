# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Model-agnostic LIBERO environment glue for the simulation/libero package.

Vendored from FastWAM's experiments/libero/libero_utils.py so this simulator package has
zero dependency on any policy/model repo. Provides just the pieces a closed-loop or
interactive harness needs: build an env for a benchmark task, pull the agentview+wrist
image, the no-op action, per-suite horizons, and the list of shipped suites.
"""
import contextlib
import logging
import os
import pathlib
import warnings

# Quiet the noisy third-party import chatter (not errors). robosuite logs a "no private
# macro file" WARNING via its logger; gym prints its "unmaintained / NumPy 2.0" notice
# straight to stderr at import time (NOT through warnings, so a filter can't catch it).
# Suppress robosuite by raising its logger to ERROR, and gym by eagerly importing it once
# with stderr redirected -- later imports (by libero/robosuite) hit the module cache and
# stay quiet. Real errors still propagate (logger level is ERROR, not CRITICAL).
for _name in ("robosuite_logs", "robosuite"):
    logging.getLogger(_name).setLevel(logging.ERROR)
with contextlib.redirect_stderr(open(os.devnull, "w")):
    try:
        import gym  # noqa: F401
    except Exception:  # noqa: BLE001
        pass
# Set the warnings filter AFTER importing gym: gym resets the warnings registry on import,
# which would otherwise wipe this filter and let robosuite's deprecated-.warn() notice leak.
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
from libero.libero import benchmark as _benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

SUITES = ["libero_object", "libero_goal", "libero_spatial", "libero_10", "libero_90"]

_SUITE_MAX_STEPS = {
    "libero_spatial": 400,
    "libero_object": 400,
    "libero_goal": 400,
    "libero_10": 700,
    "libero_90": 700,
}


def get_max_steps(task_suite_name):
    if task_suite_name not in _SUITE_MAX_STEPS:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    return _SUITE_MAX_STEPS[task_suite_name]


def get_benchmark_dict():
    return _benchmark.get_benchmark_dict()


def list_envs():
    """Enumerate every shipped (suite, task_id, description) for the env-picker dropdown.

    Cheap: reads task metadata from the benchmark registry without building any sim env.
    """
    out = []
    bench = get_benchmark_dict()
    for suite in SUITES:
        try:
            task_suite = bench[suite]()
            n_tasks = int(getattr(task_suite, "n_tasks", 0))
            for task_id in range(n_tasks):
                out.append({
                    "suite": suite,
                    "task_id": task_id,
                    "description": task_suite.get_task(task_id).language,
                })
        except Exception:  # noqa: BLE001 - skip a suite that fails to enumerate
            continue
    return out


def get_libero_env(task, resolution, seed):
    """Initialize a single OffScreenRenderEnv for a task; returns (env, description)."""
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)  # seed affects object positions even with a fixed initial state
    return env, task_description


def get_libero_dummy_action():
    """No-op action (open gripper) used to settle the sim while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def get_libero_image(obs):
    """Extract + preprocess the agentview and wrist images (rotate 180 to match training)."""
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    return {"image": img, "wrist_image": wrist_img}
