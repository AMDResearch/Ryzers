# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Model-agnostic Policy ABC for the interactive harness.

A model ships a `build_policy() -> Policy` factory, selected via POLICY_FACTORY=module:function
(default RandomPolicy). Interactive seam only; closed-loop keeps RoboTwin's eval_policy.py runner.
"""
import importlib
from abc import ABC, abstractmethod

from sim_robotwin.envutil import env_str


class Policy(ABC):
    """Turns (obs, instruction) into a [T, action_dim] chunk.

    RoboTwin action (aloha-agilex): [left_arm(6), left_grip(1), right_arm(6), right_grip(1)].
    """

    replan_steps = 8      # env steps executed per predicted chunk before replanning
    name = "policy"
    # take_action space: "qpos" (default, joint-space rows) or "ee" (absolute EE poses, RoboTwin IK).
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
        except Exception:  # noqa: BLE001
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
