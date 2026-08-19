# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Built-in model-free policy: small joint-space random walk (grippers held); default when no policy is chained."""
import numpy as np

from sim_robotwin.policy import Policy


class RandomPolicy(Policy):
    name = "random"
    replan_steps = 8

    def __init__(self, arm_scale=0.04, horizon=8, seed=0):
        self.arm_scale = float(arm_scale)
        self.horizon = int(horizon)
        self.rng = np.random.default_rng(seed)

    def predict_action_chunk(self, obs, instruction):
        vec = np.asarray(obs["joint_action"]["vector"], dtype=np.float32)
        dim = vec.shape[0]
        half = dim // 2  # [left(arm6+grip1), right(arm6+grip1)]
        chunk = np.tile(vec, (self.horizon, 1)).astype(np.float32)
        # perturb the arm joints only; keep both grippers at their current value.
        arm_idx = [i for i in range(dim) if i not in (half - 1, dim - 1)]
        deltas = self.rng.normal(0.0, self.arm_scale, size=(self.horizon, len(arm_idx)))
        deltas = np.cumsum(deltas.astype(np.float32), axis=0)
        chunk[:, arm_idx] += deltas
        return chunk


def build_policy():
    return RandomPolicy()
