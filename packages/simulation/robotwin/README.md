### RoboTwin 2.0

This package provides the RoboTwin 2.0 simulation base (SAPIEN offscreen Vulkan ray-tracing
renderer plus mplib motion planning) as a Ryzer on AMD Ryzen AI Max+ 395 (Strix Halo,
gfx1151) under ROCm 7.14. RoboTwin 2.0 is a dual-arm manipulation benchmark. The base exposes
a model-agnostic `sim_robotwin.Policy` seam, selected at runtime via `POLICY_FACTORY` (default
the built-in `RandomPolicy`), that WAM and VLA packages chain on for closed-loop and
interactive rollouts.

### Build

```sh
ryzers build simulation/robotwin
ryzers run     # test.py: ROCm torch + SAPIEN/mplib import sign-of-life
```

### Example: RoboTwin 2.0 with AHA-WAM

```sh
ryzers build simulation/robotwin ahawam
ryzers run --name ahawam-robotwin /ryzers/demos/demo_closedloop_robotwin.sh
```

<p align="center">
  <img src="assets/robotwin_ahawam.gif" width="260">
  <br><em>AHA-WAM closed-loop rollout in RoboTwin 2.0 (beat_block_hammer task).</em>
</p>

### References

- Upstream: https://github.com/RoboTwin-Platform/RoboTwin (pinned in `docs/UPSTREAM_PIN.commit.txt`)

Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
