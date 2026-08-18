# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Model-agnostic RoboTwin simulator harness.

Ships a RoboTwin task-env wrapper, a `Policy` seam, generic interactive/sanity harnesses,
and a built-in RandomPolicy. Any policy/model (FastWAM, ...) drives the interactive
showcase by providing a `build_policy() -> Policy` factory selected via POLICY_FACTORY.
The parity-critical closed-loop keeps RoboTwin's own script/eval_policy.py runner.
"""
from sim_robotwin.policy import Policy, load_policy

__all__ = ["Policy", "load_policy"]
