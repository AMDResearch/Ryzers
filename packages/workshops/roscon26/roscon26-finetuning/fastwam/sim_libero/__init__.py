# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Model-agnostic LIBERO simulator harness.

Ships the LIBERO env glue, a `Policy` seam, generic closed-loop/interactive harnesses,
and a built-in RandomPolicy. Any policy/model (FastWAM, MolmoACT2, ...) drives the sim by
providing a `build_policy() -> Policy` factory selected via POLICY_FACTORY.
"""
from sim_libero._torch_compat import patch_torch_load

patch_torch_load()

from sim_libero.policy import Policy, load_policy  # noqa: E402

__all__ = ["Policy", "load_policy"]
