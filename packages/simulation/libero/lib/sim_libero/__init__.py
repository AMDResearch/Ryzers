# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Model-agnostic LIBERO simulator harness.

LIBERO env glue, a `Policy` seam, closed-loop/interactive runners, and a RandomPolicy.
Policies plug in via a `build_policy() -> Policy` factory selected by POLICY_FACTORY.
"""
from sim_libero._torch_compat import patch_torch_load

patch_torch_load()

from sim_libero.policy import Policy, load_policy  # noqa: E402

__all__ = ["Policy", "load_policy"]
