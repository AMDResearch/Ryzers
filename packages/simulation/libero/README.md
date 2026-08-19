### LIBERO

This package provides the LIBERO simulation base (MuJoCo/EGL) as a Ryzer on AMD Ryzen AI
Max+ 395 (Strix Halo, gfx1151) under ROCm 7.14. It exposes a model-agnostic `Policy` seam
that WAM and VLA packages chain on for closed-loop and interactive rollouts.

### Build

```sh
ryzers build libero
ryzers run     # test.py: sim import + headless render sign-of-life
```

### Example: LIBERO with FastWAM

```sh
ryzers build libero fastwam --name fastwam-libero
ryzers run --name fastwam-libero /ryzers/demos/demo_closedloop_libero.sh
```

<p align="center">
  <img src="assets/libero_fastwam.gif" width="480">
  <br><em>FastWAM closed-loop rollout in LIBERO.</em>
</p>

### References

- Upstream: https://github.com/Lifelong-Robot-Learning/LIBERO (the Dockerfile shallow-clones the default branch)

Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
