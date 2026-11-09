# ROSCon 2026: Local Robot Inference

Five notebooks that put a robot under the control of a language model running
entirely on the machine in front of you. Nothing in the course calls a hosted
API, and with the default build nothing reaches the network at all.

| Notebook | What happens | Kernel |
|----------|--------------|--------|
| `0_overview.ipynb` | Where the four sections fit together | RAI (ROCm) |
| `1_local_inference.ipynb` | Lemonade serving Gemma-4 E2B on the iGPU: chat, vision, speech | RAI (ROCm) |
| `2_robot_agents.ipynb` | A RAI agent drives a Franka arm in an O3DE scene from camera images | RAI (ROCm) |
| `3_code_as_policy.ipynb` | CaP-X: the model *writes* the policy, Robosuite runs it | CaP-X (ROCm) |
| `4_robot_harness_optimization.ipynb` | HELIX evolves the policy repository; seed vs deployed, replayed live | CaP-X (ROCm) |

## Build & Run

```sh
ryzers build jupyterlab ros o3de rai lemonade-sdk capx roscon26-localinference
ryzers run /ryzers/start_roscon26_localinference.sh
```

Then open <http://localhost:8888> and work through the notebooks in order. Each
one already carries the right kernel in its metadata, so it selects itself.

Other entry points:

```sh
ryzers run                                                   # smoke test (default CMD)
ryzers run /ryzers/demo_roscon26_localinference.sh            # notebook 2 as a Streamlit page
ryzers run bash                                              # poke around
```

This ryzer is a **thin layer**. Build the whole chain - standalone it installs
the workshop patches and content but has no ROS, no CaP-X and no notebook
server to run them in. `test.sh` degrades to warnings in that case rather than
failing, so you can still see what is and is not there.

## Why this chain, in this order

| Position | Package | What it contributes |
|---|---|---|
| 1 | [`jupyterlab`](../../../ide/jupyterlab) | the notebook server and the `8888` mapping |
| 2 | [`ros`](../../../ros/ros) | ROS 2 Jazzy desktop, `cv_bridge`, `web_video_server` |
| 3 | [`o3de`](../../../graphics/o3de) | Vulkan SDK and the O3DE engine |
| 4 | [`rai`](../../../robotics/rai) | `/ryzers/rai`, the manipulation demo, GroundingDINO + SAM2 weights |
| 5 | [`lemonade-sdk`](../../../llm/lemonade-sdk) | `lemond`, the OpenAI-compatible server on `:13305` |
| 6 | [`capx`](../../../robotics/capx) | `/ryzers/cap-x`, Robosuite, PyRoKi, Contact-GraspNet |
| 7 | `roscon26-localinference` | the workshop layer described below |

Order is not cosmetic:

- `rai` has to come **before** `capx`. Its last step is
  `pip install --force-reinstall` of the ROCm torch wheels, which rebuilds the
  whole dependency tree it touches. Running that after `capx` would re-resolve
  CaP-X's pinned `numpy==1.26.4` / `mujoco==3.5.0` set.
- `ros` has to come **before** `rai`, which is also the `rai` Dockerfile's own
  default base.
- `o3de` sits between them to mirror the order the original course image used
  (ROS, then Vulkan/O3DE, then everything that renders through them).
- `lemonade-sdk` installs from a PPA and needs `software-properties-common`,
  which arrives with `ros`.

### Dropping O3DE

The workshop's simulation is the **prebuilt RAI manipulation demo binary** that
`rai`'s `download_demo.sh` fetches, not the O3DE editor. Nothing in the five
notebooks invokes `/opt/O3DE`. If image size matters more than having the
engine available to students, `o3de` can come out of the chain - `rai` already
installs the Vulkan/Mesa drivers the demo binary needs:

```sh
ryzers build jupyterlab ros rai lemonade-sdk capx roscon26-localinference
```

This has **not** been built and tested; `o3de` is in the recommended chain
because the original course image shipped it.

## What this layer adds

Everything else is composed. This package contributes exactly six things.

### 1. Headless Weston + Xwayland

O3DE renders through Vulkan, which needs a *presentable* surface. Xvfb cannot
provide one - without DRI3 only the llvmpipe software device is presentable -
but Xwayland on a headless Weston compositor can, because it hands the X server
the real `/dev/dri` render node. The camera is then republished as MJPEG by
`web_video_server` and read straight out of the notebook.

This lives here rather than in the `o3de` ryzer because the ryzer assumes a
desktop session and this workshop assumes a browser.

### 2. The CaP-X workshop patches

[`patch_capx_workshop.py`](patch_capx_workshop.py) applies four changes on top
of the composed `capx` layer. It is a patch layer and not a capx fork, so
`CAPX_REF`, the dependency resolution and the submodule selection stay owned by
[`packages/robotics/capx`](../../../robotics/capx).

| Change | Why |
|---|---|
| `FrankaControlApi` and `FrankaControlSpillWipeApi` default to `use_sam3=False` | CaP-X registers the **gated** `facebook/sam3` by default. The four live scenarios name the plain API in their `env_configs`, so flipping the default - not adding an alias - is what removes the `HF_TOKEN` requirement. Text grounding moves to OWLv2, segmentation to box-prompted SAM2. |
| FP16 checkpoint staging in `launch_owlvit_server.py` and `launch_sam2_server.py` | ROCm spends minutes transferring float32 CPU tensors to an APU. Load as float16, restore float32 on device. The GPU-side conversion is effectively instantaneous and execution stays FP32. |
| `runner.py` service wait 120 s → 900 s | Cold-starting the perception services takes longer than upstream's timeout, which fails the first trials of every run. |
| Four `env_configs` rewritten, plus `*_fast.yaml` siblings | Swaps the SAM3 server for OWLv2-large (`:8117`) + SAM2.1-large (`:8113`); the `_fast` profile carries the base/small pair for an A/B. |

The composed `capx` ryzer already raises the 900 s timeout and already appends
a `FrankaControlApiSam2` alias; the script detects both and does not fight
them. Every rewrite asserts its match count, so an upstream change fails
`docker build` instead of producing an image that only breaks in the room.

### 3. The `/opt/capx-venv` shim

The JupyterHub image kept CaP-X in its own venv, so the workshop scripts, the
recorded `opencode.json` permission rules and the evaluator command baked into
the recorded `helix.toml` all spell the interpreter `/opt/capx-venv/bin/python`:

```toml
# recorded_results/repos/seed/helix.toml
command = "/opt/capx-venv/bin/python probe.py"
```

The composed `capx` ryzer installs into the base image's interpreter instead.
Rather than rewrite that literal in a dozen places - including inside recorded
JSON that has to keep matching what the search actually ran - the Dockerfile
points the old name at `sys.prefix`:

```
/opt/capx-venv -> /opt/venv        (whatever the base image's prefix is)
```

Notebook 4's live replay runs `probe.py` through that path, so the symlink is
load-bearing, not cosmetic.

### 4. Baked weights

| What | Where | Size | Baked? |
|---|---|---|---|
| Gemma-4 E2B-it Q4_K_M + `mmproj-F16` | `/opt/lemonade-cache` | 4.1 GB | yes |
| Gemma-4 E4B-it Q4_K_M + `mmproj-F16` | `/opt/lemonade-cache` | 6.0 GB | yes |
| SAM2.1 hiera-large, OWLv2-large | `/opt/capx-cache` | 2.7 GB | yes |
| SAM2.1 hiera-small, OWLv2-base (`_fast` profile) | `/opt/capx-cache` | 0.8 GB | yes |
| GroundingDINO + SAM2 for RAI | `/root/.cache/rai` | ~1.5 GB | yes, by the `rai` ryzer |
| Qwen3-Coder-30B-A3B Q4_K_M | - | 18 GB | **no** |

**That is roughly 13.6 GB of weights added by this layer alone**, on top of
ROS 2 desktop, the O3DE deb and the CaP-X Python stack. Budget accordingly.

They are baked because a conference room is the worst possible place to
discover that 13 GB still has to come down a shared wifi link, and because
every one of them is ungated - there is no Hugging Face token to hand out.
Only the quantizations Lemonade's built-in aliases resolve to are fetched; the
GGUF repositories carry twenty-odd each. The perception repositories ship the
same checkpoints twice (`.safetensors` and `.pt`/`.bin`) and only the
safetensors copy is taken.

Both groups are opt-out, and an environment variable beats the `config.yaml`
entry:

```sh
BAKE_LEMONADE_MODELS=0 BAKE_PERCEPTION_WEIGHTS=0 \
  ryzers build jupyterlab ros o3de rai lemonade-sdk capx roscon26-localinference
```

With either set to `0` the image is that much smaller and the weights download
on first use instead. `test.sh` knows which way the image was built and warns
rather than fails.

**Qwen3-Coder-30B is deliberately not baked.** Notebook 4 replays a recorded
evolution and calls no language model at all, so 18 GB would serve one opt-in
path. The notebook prints the `lemonade pull` command that fetches and
registers it, next to the study command it feeds.

The caches live under `/opt` rather than `$HOME` on purpose. That started as a
JupyterHub workaround, but it still earns its keep: the composed `lemonade-sdk`
ryzer bind-mounts `$HOME/.cache/huggingface` and `$HOME/.cache/lemonade` from
the host, which would hide anything baked underneath them.

### 5. HELIX and OpenCode

Notebook 4's toolchain: HELIX drives repository-level evolution, OpenCode is
its local coding-agent backend. Neither has a ryzer of its own. They are
installed last so a harness bump does not rebuild the simulator, the perception
stack or 13 GB of weights.

### 6. Two Jupyter kernels

| Kernel | Display name | Environment |
|---|---|---|
| `rai` | RAI (ROCm) | ROS 2 + the `/ryzers/rai` colcon overlay, sourced by [`rai_kernel.sh`](rai_kernel.sh) |
| `capx` | CaP-X (ROCm) | `HF_HOME=/opt/capx-cache`, `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`, `CAPX_ROOT=/ryzers/cap-x` |

The names match what is already recorded in the notebooks' metadata, so each
notebook selects its own kernel on open.

In the composed image **both run the same interpreter**. The original course
needed two venvs because CaP-X pinned `numpy`, `transformers`, `matplotlib` and
`mujoco` against what the RAI stack wanted; under Ryzers that separation is
gone, because the `rai` and `capx` ryzers both install into the base image's
environment. The two kernels survive because their *environments* still differ:
CaP-X needs the baked perception cache and an EGL backend chosen before MuJoCo
imports, while RAI needs the ROS overlay. See
[Risks](#risks-and-what-was-not-verified) for what that merge could cost.

`rai_kernel.sh` sources ROS inside the kernel rather than relying on the
notebook server's environment, so a kernel restart cannot lose it. The server
still gets its own copy from `start_roscon26_localinference.sh`, because the
`!ros2 topic list` cells in notebook 2 are shell escapes that inherit the
*server's* environment, not the kernel's.

## The merged chain, resolved

Produced with the repository's own `ConfigManager` over all seven
`config.yaml` files:

**Ports** - no duplicates:

| Port | From | For |
|---|---|---|
| 8888 | `jupyterlab` | JupyterLab |
| 11311 | `ros` | ROS 1 bridge |
| 8501 | `rai` | Streamlit agent UI |
| 13305 | `lemonade-sdk` | Lemonade OpenAI-compatible API |

This package declares none. Ryzers runs with `--network=host`, so the
container-internal services need no mapping at all: `web_video_server` on
`:8080`, and the CaP-X services on `:8113` (SAM2), `:8115` (Contact-GraspNet),
`:8116` (PyRoKi) and `:8117` (OWLv2).

**Environment** - `JUPYTER_CONFIG_DIR` from `jupyterlab`, `MUJOCO_GL` and
`PYOPENGL_PLATFORM` from `capx`. This package adds none at run time; its
variables are Dockerfile `ENV`, so they also reach `docker exec`, the kernels,
and anything the notebooks spawn, which a `-e` flag does not.

**GPU / X11 / cameras** - GPU on (`--device=/dev/kfd --device=/dev/dri`,
`--group-add video --group-add render`). X11 **off** and host cameras **off**,
both set here and both last-wins across the chain. The only camera in the
course is the one O3DE simulates.

### Resolved chain conflict

Two packages in this chain map different host paths onto the same container
path:

```
-v $PWD/workspace/.cache/huggingface:/root/.cache/huggingface   # lemonade-sdk
-v ~/.cache/huggingface:/root/.cache/huggingface                # capx
```

Docker rejects two mounts on one destination with
`Error response from daemon: Duplicate mount point`, and
`configs_to_runflags()` originally appended one `-v` per entry with no
de-duplication, so `ryzers run` on this chain failed outright.

Fixed in `ryzers/configs.py`: `port_mappings` and `volume_mappings` are now
de-duplicated on the part Docker treats as unique — host port and container
path respectively — last entry winning, the same way `configs_to_buildflags()`
already resolves repeated build arguments. The package closest to the end of
the chain decides, so `capx`'s mount wins here.

This was fixed in the shared code rather than by commenting out `capx`'s mount,
which would have cost `capx` standalone a re-download of its SAM3/GraspNet
weights on every run. Neither mount is load-bearing for this workshop in any
case: the kernels and `lemonade_env.sh` point `HF_HOME` at `/opt/capx-cache`
and `/opt/lemonade-cache`.

## Pinned versions

| Pin | Value | Verified |
|---|---|---|
| HELIX | `KE7/helix@29cfa6e5eae902f6bc6d2113e51499e92c6109ee` (0.2.1) | commit resolves on GitHub |
| OpenCode | `opencode-ai@1.18.18` | version exists on the npm registry |
| CaP-X | inherited: `capgym/cap-x@53e9966d7a8e2fa7494676772bccc35280f5c0ed` | commit resolves; all patch anchors verified against it |
| RAI | inherited: `RobotecAI/rai@e54f8ca` | commit resolves; `examples/manipulation_common.py`, `examples/embodiments/manipulation_embodiment.json` and `config.toml` all present |
| ROS 2 | inherited: `jazzy` | - |
| `ipywidgets` | `8.1.9` | exists on PyPI |
| `pandas` | `>=2.2,<3` | ranged on purpose - nothing here has been exercised against pandas 3 |
| Gemma-4 E2B/E4B GGUF, SAM2.1 large/small, OWLv2 large/base | see table above | all six repositories resolve and the exact file names were checked |

`streamlit` is deliberately absent: `rai_core` already declares
`streamlit>=1.44,<2` and owns that pin.

Notebook-only Python additions install with
`-c /ryzers/constraints.txt` - the constraints file the composed `capx` layer
writes, carrying the base torch plus CaP-X's `numpy==1.26.4` override - so a
convenience package cannot drag numpy to 2.x and take `rai_perception`'s
numpy-1-ABI scikit-learn with it.

## Notebook 4's recorded study

`recorded_results/` ships in the image at
`/ryzers/notebooks/recorded_results`. The upstream course README says it is not
committed; **that is stale** - it is tracked, 25 files, 232 KB, and it
validates:

```
[OK] study report      rho_multitask_report.json   (schema rho-multitask-helix-report/v2)
[OK] seed repository   repos/seed/solver
[OK] selected          repos/selected/solver
[OK] seed helix.toml / probe.py / opencode.json
```

`test.sh` runs `rho_report.preflight()` on it, so a truncated copy fails the
build's smoke test rather than the session. The recorded run is a
two-generation, two-task (`cube_stack`, `cube_lift`) Qwen3-Coder evolution,
47.5 minutes of study time, deploying candidate `g1-s3`.

`videos/` is **not** included, by design: the notebook replays both policies
live instead of rendering recorded clips, and it is the only large thing the
study produces.

To regenerate it, notebook 4 prints both commands - the `lemonade pull` for the
30B mutation model and the `rho_multitask_study.py` invocation - or see
`scripts/rho_report.py:REGENERATE_COMMAND`.

## Checks

```sh
ryzers run                                        # fast: this layer only, no services
ryzers run /ryzers/test_ros.sh                    # from the composed ros ryzer
ryzers run /ryzers/test_o3de.sh                   # from the composed o3de ryzer
ryzers run /ryzers/test_rai.sh                    # from the composed rai ryzer
ryzers run /ryzers/test_capx.sh                   # from the composed capx ryzer
ryzers run /ryzers/test_lemonade-sdk.sh           # from the composed lemonade ryzer

ryzers run /ryzers/test_roscon26_lemonade.sh      # workshop: baked Gemma + a completion
ryzers run /ryzers/test_roscon26_capx.sh          # workshop: oracle eval against the patches
ryzers run /ryzers/test_roscon26_rho.sh           # workshop: HELIX harness
ryzers run /ryzers/test_roscon26_rho_multitask.sh
```

The `roscon26_`-prefixed ones are the course's own deeper checks. They are
prefixed because `/ryzers/test_capx.sh`, `/ryzers/test_rai.sh`,
`/ryzers/test_ros.sh` and `/ryzers/test_lemonade-sdk.sh` already belong to the
composed ryzers - overwriting them would trade the upstream checks for the
workshop ones instead of keeping both.

## Measured workshop runtime

From the original course, on the workshop Strix Halo GPU. Treat as indicative;
none of it was re-measured on this image.

- CaP-X: 4.5 s model setup, 11.4 s perception/control setup, 14.2 s for one LLM
  call, 11.8 s for one rollout.
- RHO service startup: the `ensure_services` cell sits immediately above the
  live replay, so nothing loads OWLv2, SAM2, Contact-GraspNet or PyRoKi until
  the notebook needs them. It is idempotent - run it early if you would rather
  pay the cost before the session.
- RHO seed-vs-deployed replay: four rollouts, two per task, 10-16 s each. The
  only live compute in notebook 4. The seed crashes every time; measured five
  times on each of the six validation trials, the deployed policy solved 19 of
  30, so budget for the occasional miss rather than treating it as a failure.
- Notebook 2's agent run: four to five minutes, cold.

Every score notebook 4 shows comes from the six validation trials the search
optimised against, which keeps the session short at the cost of saying nothing
about generalisation. The notebook states that where it reports the numbers.

## Risks and what was not verified

**Built and smoke-tested on the target: Ryzen AI MAX+ 395 / Radeon 8060S
(gfx1151)**, and on an RX 7900 XTX. The full seven-package chain builds, the
smoke test passes with zero warnings, and `torch.cuda.is_available()` is true
with a GPU matmul running on the 8060S.

**Resolved by building — no longer open:**

- **One interpreter instead of two works.** `rai` and `capx` both import in the
  base environment, on torch `2.12.0+rocm7.14.0`, numpy `1.26.4` and
  transformers `4.57.6`. The `cv2` ownership worry is moot: `opencv-python` and
  `opencv-python-headless` both land at the identical `4.11.0.86`.
- **Chain order matters, as predicted.** torch drops to `2.3.1` partway through
  the chain and is restored by `rai`'s `--force-reinstall` — which is exactly
  why `rai` must precede `capx`. Reordering them would leave the wrong torch.
- **The duplicate HF cache mount is fixed upstream of this package**, by adding
  port/volume dedup to `configs_to_runflags()` rather than editing `capx`.
- **The CaP-X workshop patches apply** over the composed `capx` layer: registry
  default, FP16 staging, 900 s wait and 4 scenario configs (+ `_fast`).
- **Both kernelspecs register**, and the `/opt/capx-venv` shim resolves such
  that `/opt/capx-venv/bin/python` imports `capx` and ROCm torch — which is what
  notebook 4's recorded `helix.toml` invokes.
- **Image size measured:** 105 GB on x86 RDNA 3, 168 GB on the Strix Halo box.

**Still open:**

1. **`dtype=` on `from_pretrained` / `pipeline`.** The FP16 staging patch uses
   the `dtype=` keyword. It patched cleanly and the files compile, but no
   perception server was actually started, so the keyword is unexercised.
2. **Weston on the target host.** The headless compositor needs a working
   `/dev/dri` render node inside the container. The run flags pass it through,
   but Weston's GL renderer was never initialised.
3. **No notebook was run.** No cell, service, rollout or RHO replay has been
   executed — only the layer smoke test. Every timing and success rate quoted
   in this README comes from the original course image, not from a run here.

## Dropped from the original course image

- `jovyan` user, `HOST_RENDER_GID`, `groupadd hostrender`, the `chown -R`
  layering dance and `/tmp/runtime-jovyan`. Ryzers runs as root and adds
  `--group-add render/video` at run time.
- `jupyter-server-proxy`. It existed to route the Streamlit UI and camera
  stream through each JupyterHub user's notebook route; with `--network=host`
  you reach `:8501` and `:8080` directly.
- The `/opt/rai-cache` rewrite of `rai_perception`'s weight path. That existed
  because JupyterHub mounted a PVC over `$HOME`; here `/root/.cache/rai` is in
  the image and nothing mounts over it, so the composed `rai` ryzer's default
  is correct.
- The separate `/opt/rai-venv` and `/opt/capx-venv` virtualenvs - see
  [Two Jupyter kernels](#6-two-jupyter-kernels) and the shim above.
- `tests/test_ros.sh` and `tests/test_o3de.sh` from the course tree, which
  duplicated what the composed ryzers already install.
- `tests/test_rai.sh`, whose assertions were specific to `/opt/rai-cache` and
  `jupyter-server-proxy`. Its still-relevant checks (the demo scripts, the RAI
  example assets, the colcon overlay) moved into this package's `test.sh`.
- The `start-local-inference.sh` entrypoint, replaced by
  `/ryzers/start_roscon26_localinference.sh` with the `jovyan` parts removed.

## References

- Lemonade: <https://lemonade-server.ai/>
- RAI: <https://github.com/RobotecAI/rai>
- CaP-X: <https://github.com/capgym/cap-x>
- O3DE: <https://o3de.org/>
- Code as Policies: <https://code-as-policies.github.io/>
- SAM 2: <https://github.com/facebookresearch/sam2>
- OWLv2: <https://huggingface.co/docs/transformers/model_doc/owlv2>
- Contact-GraspNet: <https://github.com/NVlabs/contact_graspnet>
- PyRoKi: <https://github.com/chungmin99/pyroki>

Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
