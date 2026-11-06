# ROSCon 2026: Reinforcement Learning

Watch a Brax PPO policy get better at `PandaPickCube` — a Franka arm learning to
reach, grasp, and lift a cube in MuJoCo. The notebook rolls out three saved
checkpoints from the same training run (early, mid, final) under an identical
seed and renders each one to video, so the only thing that changes between the
clips is how much the policy has learned.

This ryzer does **not** train. All three checkpoints ship in the image and the
notebook runs deterministic inference only.

## Build & Run

```sh
ryzers build jupyterlab roscon26-rllearning
ryzers run "jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
```

Then open `0_simulation_rl.ipynb` and pick the **ROSCon26 RL (ROCm)** kernel.

JupyterLab and its `8888` port mapping come from the composed
[`jupyterlab`](../../../ide/jupyterlab) ryzer, so build the chain rather than
this package alone. Building it standalone still works — the test warns instead
of failing — but there is no notebook server to open.

```sh
ryzers run                                   # smoke test (default CMD)
ryzers run /ryzers/demo_roscon26_rllearning.sh final   # one rollout + mp4
```

## What is in the image

| Path | Contents |
|------|----------|
| `/ryzers/notebooks/0_simulation_rl.ipynb` | The workshop notebook |
| `/ryzers/notebooks/assets/PandaPickCube-*/` | Three Brax PPO checkpoints, unzipped at build |
| `/ryzers/notebooks/assets/headless_gl.py` | Offscreen GL backend probing |
| `/ryzers/notebooks/scripts/render_trajectory.py` | Re-render a saved rollout without rerunning inference |
| `/ryzers/notebooks/scripts/render_checkpoint_once.py` | One-off rollout + render for a single checkpoint |
| `/opt/rl-venv` | JAX/MuJoCo/Brax stack, exposed as the `roscon26-rl` kernel |

Checkpoint progression, in the order the notebook runs them:

| Stage | Checkpoint | Video |
|-------|-----------|-------|
| Early — weak | `20260807-131132/checkpoints/000008192000` | `panda_pick_cube_early.mp4` |
| Mid — improving | `20260817-150103/checkpoints/000006553600` | `panda_pick_cube_mid.mp4` |
| Final — strong | `20260817-150103/checkpoints/000045875200` | `panda_pick_cube.mp4` |

Stages are ordered by rollout quality rather than step count alone, which is why
the "mid" checkpoint has a lower step number than the "early" one — they come
from two different training runs.

## Why the RL stack lives in its own venv

ROCm JAX loads its HIP libraries from the `rocm` **wheel** SDK (7.13), while the
base image's PyTorch uses the system ROCm at `/opt/rocm` (7.14). Putting both in
one `site-packages` would place two ROCm runtimes on a single loader path, so
the JAX stack gets a standalone venv at `/opt/rl-venv` and a dedicated
kernelspec. Composing this ryzer with a torch-based package therefore leaves
that torch untouched. The same reasoning applies to the
[`capx`](../../../robotics/capx) image's separate environments.

## ROCm JAX version pins

`jax`, `jaxlib`, `jax-rocm7-plugin` and `jax-rocm7-pjrt` are pinned to **0.11.1**
from PyPI, with `rocm[libraries]==7.13.0` from AMD's per-architecture wheel index
(`https://repo.amd.com/rocm/whl/gfx1151/`).

That split is deliberate:

- The gfx1151 index carries `jax-rocm7-*` only up to 0.9.1, and jax 0.9.x hits
  `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` in a fused scatter on this chip.
  The underlying XLA fix landed for jax >= 0.10.
- 0.11.1 is the combination reported working on Ryzen AI MAX+ 395 / Radeon 8060S.
- `brax==0.14.2` declares `jax>=0.4.6` with no upper bound, so the RL stack
  resolves against the ROCm jax rather than pulling a CPU `jaxlib` over it.

That last point needs enforcing rather than trusting. An unbounded `jax>=…`
lets pip upgrade `jax`/`jaxlib` from PyPI while `jax-rocm7-plugin` stays pinned,
leaving the set out of lockstep and the GPU silently unused. So the ROCm set is
frozen into a constraints file applied to every later install, and the build
asserts all four packages still report the same version. The sibling
[`roscon26-simulation`](../roscon26-simulation) hit exactly this failure with
torch, where it replaced the ROCm build with a CUDA one on one architecture
only.

### Verification status

**Verified on the target: Ryzen AI MAX+ 395 / Radeon 8060S (gfx1151.)** The
default `GPU_TARGET` builds cleanly, `jax.devices()` reports
`[RocmDevice(id=0)]`, and both rollouts run on the GPU — reward **656.794** for
the early checkpoint against **1325.485** for the final one, 150 steps each,
rendered to mp4 at 50 fps.

Also verified on an RX 7900 XTX (`GPU_TARGET=gfx110X-all`). The two agree
closely: the early rollout matches to the digit, and the final differs only in
the fifth significant figure (1325.485 vs 1325.412) — ordinary floating-point
variation between architectures, not a behavioural change.

Architecture enters the image through exactly one package. Both builds install
an identical `jax`/`jaxlib`/`jax-rocm7-plugin`/`jax-rocm7-pjrt` 0.11.1 set plus
`rocm-sdk-core==7.13.0` — all architecture-agnostic, straight from PyPI — and
differ only in `rocm-sdk-libraries-<arch>`, so retargeting is a one-wheel swap.

On an APU, XLA logs `rsmi_dev_gpu_metrics_info_get failed … Assuming PCIe Gen4
x16` at startup. That is benign — there is no discrete PCIe link to measure —
and does not affect execution.

Note the Menagerie prefetch layer logs `No AMD GPUs were found, skipping ROCm
plugin initialization` during `docker build`. That is expected — `docker build`
gets no device passthrough — and is harmless, as the step only needs CPU.

`GPU_TARGET` selects the wheel bucket and defaults to the workshop's Strix Halo
target. It names a *bucket* directory under `https://repo.amd.com/rocm/whl/`,
not a raw gfx target — an RX 7900 XTX is `gfx1100` but builds against
`gfx110X-all`, and there is no `gfx1100/` directory. Use `gfx110X-all` rather
than `gfx110X-dgpu`: the latter ships the ROCm SDK but no `jax-rocm7-*` wheels.

`configs_to_buildflags()` lets a matching environment variable override a
`build_arguments` entry, so you can retarget without editing
[`config.yaml`](config.yaml):

```sh
GPU_TARGET=gfx110X-all ryzers build jupyterlab roscon26-rllearning
```

If JAX cannot reach the GPU it falls back to CPU and the notebook still
completes — rollouts are short and deterministic — so a driver mismatch degrades
rather than blocks the workshop. `test.sh` prints the active backend either way.

## Rendering

Rendering is offscreen and CPU-side: `headless_gl.py` probes `osmesa` first and
then three EGL configurations, all through llvmpipe software rasterisation, and
every rollout renders in a subprocess because the dynamic loader fixes its
search path at process start. This path is independent of ROCm, which is why
`x11_display` is disabled in [`config.yaml`](config.yaml) and no X socket is
needed.

## References

- MuJoCo Playground: <https://github.com/google-deepmind/mujoco_playground>
- MuJoCo Menagerie (Franka model): <https://github.com/google-deepmind/mujoco_menagerie>
- Brax: <https://github.com/google/brax>
- Install JAX for ROCm: <https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/jax/install.html>

Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
