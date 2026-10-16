# ROSCon 2026: Simulators

Take a Unitree G1 humanoid that can already walk on flat ground and teach it to
climb stairs, in [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) via
the `gslab` task library. The training notebook fine-tunes against a terrain
curriculum under a fixed time budget; the evaluation notebook puts the
flat-ground baseline and the stair-trained policy through the same scenarios and
renders both so the failure mode is visible rather than just a number.

## Build & Run

```sh
ryzers build jupyterlab roscon26-simulation
ryzers run jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Open `notebooks/00_workshop_intro.ipynb` first and select the **gslab ROSCon
(ROCm 7.13)** kernel. Then run `humanoid_locomotion_train.ipynb`, then
`humanoid_locomotion_eval.ipynb`.

JupyterLab and its `8888` port mapping come from the composed
[`jupyterlab`](../../../ide/jupyterlab) ryzer, so build the chain rather than
this package alone. Building it standalone still works — the test warns instead
of failing — but there is no notebook server to open.

```sh
ryzers run        # smoke test (default CMD)
```

## What is in the image

Everything lands under `/ryzers/notebooks`, which is the **workshop root**:

| Path | Contents |
|------|----------|
| `INSTALL.md` | Upstream bare-metal install guide — also the root marker (see below) |
| `checkpoints/g1_flat_baseline.pt` | Flat-ground policy the workshop starts from |
| `checkpoints/g1_stairs_easy_trained.pt` | A finished 25-minute training run, for attendees whose own run is not there yet |
| `checkpoints/g1_ref.pt` | The reference: 2,048 robots × 24 steps × 200 PPO updates |
| `notebooks/` | The three notebooks plus `workshop_helpers.py` / `workshop_training.py` |
| `link_rocm_tree.sh` | Builds the ROCm tree Genesis needs (run at image build) |
| `/opt/gslab-venv` | torch + Genesis + gslab, exposed as the `gslab-roscon` kernel |

The notebooks write training output to `logs/` under the workshop root.

### Why the nesting

`workshop_helpers.find_workshop_root()` walks up from the notebook's working
directory looking for a folder that contains **both `INSTALL.md` and
`checkpoints/`**. That makes `INSTALL.md` load-bearing rather than decorative,
and it means the notebooks must live one level down. Hence `workshop/` in this
package maps to `/ryzers/notebooks`, with the `.ipynb` files at
`/ryzers/notebooks/notebooks/`. Deleting `INSTALL.md` would break the notebooks.

## Why the stack lives in its own venv

Two ROCm layers are in play and neither can be dropped:

- **torch** finds the HIP runtime through the pip `rocm_sdk` packages its wheel
  pulls from AMD's per-architecture index.
- **Genesis** compiles physics kernels at runtime with `quadrants`, which wants
  a classic `/opt/rocm` layout the pip SDK does not ship.

`gslab` also needs torch >= 2.10, newer than the base image's. So the workshop
gets `/opt/gslab-venv` with its own torch and a dedicated kernelspec, leaving
the base image's torch untouched — the same isolation
[`roscon26-rllearning`](../roscon26-rllearning) uses for ROCm JAX.

### Pinning torch against silent replacement

`gslab` only requires `torch>=2.10`, and the layer that installs it resolves
across PyPI as well as the AMD index. pip is therefore free to satisfy that
bound with a newer **CUDA** torch from PyPI, silently replacing the ROCm build
and pulling in several GB of `nvidia-*` wheels. This is not hypothetical: it
happened here on one architecture while another, same Dockerfile, happened to
resolve correctly — so it fails intermittently rather than reliably.

The fix is a constraints file frozen from the ROCm torch immediately after it is
installed, applied via `PIP_CONSTRAINT` to every later `pip install`. A conflict
then surfaces as a resolution error at build time instead of a CPU-only
container at workshop time. Each such layer also re-asserts `torch.version.hip`,
so a regression cannot reach the image.

`link_rocm_tree.sh` bridges the gap by assembling `/opt/gslab-venv/rocm`:
unversioned `.so` aliases so `dlopen("libamdhip64.so")` resolves by bare name
(wheels cannot carry symlinks, so the SDK ships only `libamdhip64.so.7`), and an
`ld.lld` of LLVM >= 19 — Ubuntu 24.04's stock lld-18 rejects the AMDGPU v6 code
objects Genesis emits with "unknown abi version". The script prefers
`/opt/rocm/llvm/bin/ld.lld` from the base image, which is LLD 23, and verifies
both before exiting, so the build fails loudly if that ever regresses.

## Architecture targeting

`GPU_TARGET` selects both the torch wheel and the `rocm-sdk-libraries` package,
and defaults to the workshop's Strix Halo target. It names a *bucket* directory
under `https://repo.amd.com/rocm/whl/`, not a raw gfx target — an RX 7900 XTX is
`gfx1100` but builds against `gfx110X-all`, and no `gfx1100/` directory exists.

```sh
GPU_TARGET=gfx110X-all ryzers build jupyterlab roscon26-simulation
```

The kernelspec's `LD_LIBRARY_PATH` is derived at build time from the
`rocm/lib-<target>` directory that actually got created, rather than assuming
`lib-gfx1151`, so retargeting needs no further edits.

## Verification status

**Verified on the target: Ryzen AI MAX+ 395 / Radeon 8060S (gfx1151).** Genesis
initialised on the `gs.amdgpu` backend and completed a simulation step, which
means `quadrants` JIT-compiled its kernels for gfx1151 and linked them through
`ld.lld` — the part most likely to break. Also confirmed there: torch
`2.11.0+rocm7.13.0` reporting HIP, Genesis 1.4.0, `dlopen("libamdhip64.so")` by
bare name, the nine `gslab` tasks registering, `lib-gfx1151` derived into the
kernelspec, and `find_workshop_root()` resolving with all three checkpoints
present. The APU reports 64 GiB of unified device memory.

Also verified on an RX 7900 XTX (`GPU_TARGET=gfx110X-all`), which exercised the
same path and correctly derived `lib-gfx110X_all` rather than a hardcoded
`lib-gfx1151`.

Architecture is confined to the two wheels `GPU_TARGET` selects — the torch
build and `rocm-sdk-libraries-<target>` — and the kernelspec derives its
`LD_LIBRARY_PATH` from whichever `rocm/lib-*` directory those produce. So
retargeting is a wheel swap with no further edits.

## Differences from the upstream course

- Runs as root, so the `jovyan` user, host-render-group joining and the
  "keep caches outside `/home/jovyan`" workarounds are gone. Those existed only
  because JupyterHub mounts a per-user volume over the home directory.
- The venv is built with `python3 -m venv` on the base image's Python 3.12
  rather than `uv` on 3.13. `uv` is not in the Ryzers base, and the pinned torch
  publishes a cp312 wheel, so the extra toolchain earns nothing here.
- The bundled `gslab` wheel is deleted in the same layer that installs it.

## References

- Genesis: <https://github.com/Genesis-Embodied-AI/Genesis>
- MuJoCo Menagerie (Unitree G1): <https://github.com/google-deepmind/mujoco_menagerie>
- ROCm wheel index: <https://repo.amd.com/rocm/whl/>

Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
