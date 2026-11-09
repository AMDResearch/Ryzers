# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Model-agnostic policy interface for the LIBERO simulator harness.

A policy plugs into the closed-loop / interactive harness by implementing this ABC. The
harness owns the env, rendering, streaming and the episode loop; the policy only turns an
observation + instruction into an action chunk. Any model (FastWAM, MolmoACT2, VLA-JEPA,
...) ships a factory `build_policy() -> Policy` and is selected at runtime via the
`POLICY_FACTORY=module:function` env var (default: the built-in RandomPolicy).
"""
import importlib
from abc import ABC, abstractmethod

from sim_libero.envutil import env_str


class Policy(ABC):
    """Turns (obs, instruction) into a [T, action_dim] chunk. Harness executes it."""

    # LIBERO OSC_POSE control cadence knobs the harness reads (a model may override).
    replan_steps = 5      # env steps executed per predicted chunk before replanning
    num_steps_wait = 5    # no-op settle steps at episode start
    name = "policy"

    def reset(self, instruction):
        """Called once per episode before the first prediction (clear caches, etc.)."""

    @abstractmethod
    def predict_action_chunk(self, obs, instruction):
        """Return an ndarray of shape [T, 7] (dx,dy,dz,droll,dpitch,dyaw, gripper)."""

    def warmup(self, obs, instruction):
        """Optional one-time forward so the first real episode isn't stalled."""
        try:
            self.predict_action_chunk(obs, instruction)
        except Exception:  # noqa: BLE001 - warmup is best-effort
            pass


def load_policy():
    """Instantiate the policy named by POLICY_FACTORY=module:function (default RandomPolicy)."""
    spec = env_str("POLICY_FACTORY", "sim_libero.random_policy:build_policy")
    if ":" not in spec:
        raise ValueError(f"POLICY_FACTORY must be 'module:function', got {spec!r}")
    module_name, fn_name = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), fn_name)
    policy = factory()
    if not isinstance(policy, Policy):
        raise TypeError(f"{spec} did not return a sim_libero.Policy (got {type(policy)})")
    return policy


__all__ = ["Policy", "load_policy"]
