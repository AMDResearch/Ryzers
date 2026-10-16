# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""LIBERO scene wrapper for the interactive/sanity harness (model-agnostic).

Builds a single OffScreenRenderEnv for a (suite, task_id) via the vendored glue and
exposes the scene's native instruction, initial state, and object list. No policy or
model code here.
"""
import numpy as np

from sim_libero._torch_compat import patch_torch_load
from sim_libero.libero_env import (
    LIBERO_ENV_RESOLUTION,
    get_benchmark_dict,
    get_libero_env,
    get_libero_image,
)


class Scene:
    def __init__(self, suite, task_id, seed=1000, resolution=LIBERO_ENV_RESOLUTION):
        self.suite = suite
        self.task_id = int(task_id)
        self.seed = int(seed)
        self.resolution = int(resolution)

        # Re-assert the torch.load shim at the point of use: LIBERO's get_task_init_states() loads
        # numpy-pickled init states, which torch>=2.6 rejects under the new weights_only=True
        # default. The package-level patch can be clobbered by heavy model-load imports that run
        # before a scene is built (e.g. loading the FastWAM policy), so patch again right here.
        patch_torch_load()

        benchmark_dict = get_benchmark_dict()
        task_suite = benchmark_dict[suite]()
        self.task = task_suite.get_task(self.task_id)
        self.init_states = task_suite.get_task_init_states(self.task_id)
        self.env, self.description = get_libero_env(self.task, self.resolution, self.seed)
        self.objects = self._list_objects()

    def _list_objects(self):
        """Best-effort manipulable-object names for the viewport panel.

        Tries the benchmark task's declared objects of interest, then falls back to parsing
        the task's BDDL `(:objects ...)` block (instance names like `akita_black_bowl_1`,
        normalised to `akita black bowl`). Cosmetic only, so any failure yields [].
        """
        try:
            names = list(getattr(self.task, "object_of_interest", []) or [])
            if names:
                return self._clean_names(names)
        except Exception:
            pass
        try:
            import pathlib
            import re

            from sim_libero.libero_env import get_libero_path

            bddl = (pathlib.Path(get_libero_path("bddl_files"))
                    / self.task.problem_folder / self.task.bddl_file)
            block = re.search(r"\(:objects(.*?)\)", bddl.read_text(), re.S)
            names = []
            if block:
                for line in block.group(1).splitlines():
                    line = line.strip()
                    if not line or line.startswith(";"):
                        continue
                    names.extend(line.split(" - ")[0].split())
            if names:
                return self._clean_names(names)
        except Exception:
            pass
        return []

    @staticmethod
    def _clean_names(names):
        import re
        seen, out = set(), []
        for n in names:
            base = re.sub(r"_\d+$", "", str(n)).replace("_", " ").strip()
            if base and base not in seen:
                seen.add(base)
                out.append(base)
        return out

    def reset(self):
        """Reset to the task's first initial state; returns the raw obs dict."""
        self.env.reset()
        idx = 0 if len(self.init_states) else None
        obs = self.env.set_init_state(self.init_states[idx]) if idx is not None else self.env.reset()
        return obs

    def view(self, obs):
        return get_libero_image(obs)

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


def build_scene(suite, task_id, seed=1000, resolution=LIBERO_ENV_RESOLUTION):
    return Scene(suite, task_id, seed=seed, resolution=resolution)
