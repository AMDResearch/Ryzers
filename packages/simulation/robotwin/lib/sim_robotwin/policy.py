# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Model-agnostic policy interface for the RoboTwin simulator harness.

A policy plugs into the interactive harness by implementing this ABC. The harness owns
the RoboTwin task env, rendering, streaming and the episode loop; the policy only turns an
observation + instruction into a joint-space action chunk. Any model (FastWAM, ...) ships
a factory `build_policy() -> Policy` and is selected at runtime via the
`POLICY_FACTORY=module:function` env var (default: the built-in RandomPolicy).

Note: this seam drives the *interactive* showcase. The parity-critical closed-loop keeps
RoboTwin's own script/eval_policy.py + policy/<name>/deploy_policy.py runner unchanged.
"""
import importlib
from abc import ABC, abstractmethod

from sim_robotwin.envutil import env_str


class Policy(ABC):
    """Turns (obs, instruction) into a [T, action_dim] joint-space chunk.

    RoboTwin's action is per-arm qpos + gripper, concatenated:
    [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)] for aloha-agilex.
    The harness executes each row via TASK_ENV.take_action(row, action_type="qpos").
    """

    replan_steps = 8      # env steps executed per predicted chunk before replanning
    name = "policy"
    # How the harness executes each predicted row via RoboTwin's TASK_ENV.take_action:
    #   "qpos" (default) -> rows are joint-space [left_arm(6),left_grip(1),right_arm(6),right_grip(1)]
    #   "ee"             -> rows are absolute end-effector poses (RoboTwin runs its own IK planner)
    # A qpos policy (FastWAM/random) needs no change; an EE-space policy (X-WAM) sets "ee".
    action_type = "qpos"

    def reset(self, instruction):
        """Called once per episode before the first prediction (clear caches, etc.)."""

    @abstractmethod
    def predict_action_chunk(self, obs, instruction):
        """Return an ndarray of shape [T, action_dim] (qpos rows, or EE-pose rows if action_type='ee')."""

    def warmup(self, obs, instruction):
        """Optional one-time forward so the first real episode isn't stalled."""
        try:
            self.predict_action_chunk(obs, instruction)
        except Exception:  # noqa: BLE001 - warmup is best-effort
            pass


def load_policy():
    """Instantiate the policy named by POLICY_FACTORY=module:function (default RandomPolicy)."""
    spec = env_str("POLICY_FACTORY", "sim_robotwin.random_policy:build_policy")
    if ":" not in spec:
        raise ValueError(f"POLICY_FACTORY must be 'module:function', got {spec!r}")
    module_name, fn_name = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), fn_name)
    policy = factory()
    if not isinstance(policy, Policy):
        raise TypeError(f"{spec} did not return a sim_robotwin.Policy (got {type(policy)})")
    return policy


__all__ = ["Policy", "load_policy"]
