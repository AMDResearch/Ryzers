# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Environment sign-of-life for the LIBERO simulator base image (Strix Halo, gfx1151).

Runs inside the built image with NO model. Proves the container has (1) the LIBERO /
MuJoCo / robosuite stack importing cleanly and (2) the sim_libero harness + built-in
RandomPolicy resolving, before a policy image is chained on top. Does not render (that
needs a GPU device, exercised by demo_sim_sanity.sh at run time). Exits non-zero on any
failure so `ryzers run` / CI catches a broken image early.
"""
import sys


def main() -> int:
    import mujoco
    import robosuite  # noqa: F401
    from libero.libero import benchmark  # noqa: F401
    from libero.libero.envs import OffScreenRenderEnv  # noqa: F401

    import sim_libero  # noqa: F401
    from sim_libero.libero_env import SUITES, get_benchmark_dict, get_max_steps
    from sim_libero.policy import Policy, load_policy

    policy = load_policy()  # default: built-in RandomPolicy
    if not isinstance(policy, Policy):
        print("FAIL: default policy is not a sim_libero.Policy", file=sys.stderr)
        return 1

    benchmark_dict = get_benchmark_dict()
    for suite in SUITES:
        if suite not in benchmark_dict:
            print(f"FAIL: suite {suite} missing from LIBERO benchmark dict", file=sys.stderr)
            return 1
        get_max_steps(suite)

    print(f"mujoco           : {mujoco.__version__}")
    print(f"suites           : {', '.join(SUITES)}")
    print(f"default policy   : {getattr(policy, 'name', type(policy).__name__)}")
    print("deps import ok   : mujoco, robosuite, libero (envs), sim_libero + RandomPolicy")
    print("PASS: LIBERO simulator env OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
