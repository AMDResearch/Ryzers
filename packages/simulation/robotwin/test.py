# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Environment sign-of-life for the RoboTwin simulator base image (Strix Halo, gfx1151).

Runs inside the built image with NO model. Proves the container has (1) the SAPIEN / mplib
stack importing cleanly on ROCm and (2) the de-vendored RoboTwin + sim_robotwin harness +
built-in RandomPolicy resolving, before a policy image is chained on top. Does not render
(that needs a GPU device, exercised by demo_sim_sanity.sh at run time). Exits non-zero on
any failure so `ryzers run` / CI catches a broken image early.
"""
import os
import sys


def main() -> int:
    import torch
    import sapien
    import mplib.planner  # noqa: F401
    from mplib.sapien_utils import SapienPlanner  # noqa: F401

    import sim_robotwin  # noqa: F401
    from sim_robotwin.policy import Policy, load_policy

    if not torch.version.hip:
        print(f"FAIL: expected a ROCm torch build, got {torch.__version__}", file=sys.stderr)
        return 1

    robotwin_root = os.environ.get("ROBOTWIN_ROOT", "/opt/RoboTwin")
    for rel in ("envs", "script/eval_policy.py", "policy"):
        if not os.path.exists(os.path.join(robotwin_root, rel)):
            print(f"FAIL: de-vendored RoboTwin missing {rel} under {robotwin_root}", file=sys.stderr)
            return 1

    policy = load_policy()  # default: built-in RandomPolicy
    if not isinstance(policy, Policy):
        print("FAIL: default policy is not a sim_robotwin.Policy", file=sys.stderr)
        return 1

    print(f"torch            : {torch.__version__} (hip {torch.version.hip})")
    print(f"sapien           : {sapien.__version__}")
    print(f"robotwin root    : {robotwin_root}")
    print(f"default policy   : {getattr(policy, 'name', type(policy).__name__)}")
    print("deps import ok   : torch(hip), sapien, mplib, RoboTwin, sim_robotwin + RandomPolicy")
    print("PASS: RoboTwin simulator env OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
