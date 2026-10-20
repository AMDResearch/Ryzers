# Genesis Docker Setup

This folder contains the files to build and run a Docker container with [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) 1.4.x (`genesis-world`) installed using Ryzers, on top of the default ROCm PyTorch base image.

## Build & Run

```sh
ryzers build genesis
ryzers run
```

The default test is a smoke test: it checks that a HIP device is visible, initializes Genesis on the `amdgpu` backend, builds a small Franka Panda scene and steps it.

## Backends

Select the Genesis backend with the `GENESIS_BACKEND` environment variable (`amdgpu` is the default):

| Value    | Notes |
|----------|-------|
| `amdgpu` | ROCm GPU backend (experimental upstream). Verified on the Radeon 8060S iGPU (gfx1151). The first run compiles kernels, later runs use the cache in `~/.cache/quadrants`. |
| `cpu`    | Works everywhere, no GPU needed. |
| `gpu`    | amdgpu if available, otherwise falls back to CPU. |

`gs.vulkan` (used by Genesis 0.3.x) no longer exists in Genesis 1.x.

ROCm-specific settings baked into the Dockerfile (see the comments there for details):
- Genesis is installed from a pinned main-branch commit instead of the `genesis-world==1.4.4` wheel, because the wheel fails in `scene.build()` on AMD GPUs (Genesis issue #3544, fixed by PR #3545).
- `LD_PRELOAD` of torch's `libamdhip64` so that Quadrants and PyTorch share one HIP runtime.
- `GS_ENABLE_ZEROCOPY=0`: zero-copy torch/Quadrants tensor sharing segfaulted in `scene.build()` on the ROCm 7.14 image.
- `QD_PERFDISPATCH_FORCE=func_solve_body:func_solve_body_monolith` forces the monolithic constraint solver, since the default decomposed one did not finish compiling on gfx1151.

## Demo (Tactile sensors on a Franka gripper)

```sh
ryzers build genesis
ryzers run /ryzers/demo_genesis.sh
```

Add `--vis` for the interactive viewer (needs X11), and `--sensor {elastomer,depth,kinematic,proximity}` to pick the sensor. The script runs
https://github.com/Genesis-Embodied-AI/Genesis/blob/v1.4.4/examples/sensors/tactile_franka.py
To run on the CPU, add `- "GENESIS_BACKEND=cpu"` to `environment_variables` in this package's `config.yaml` and rebuild.

## References
- https://genesis-world.readthedocs.io/en/latest/
- https://github.com/Genesis-Embodied-AI/Genesis

Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
