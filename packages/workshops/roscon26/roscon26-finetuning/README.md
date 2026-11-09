# ROSCon 2026: Fine-tuning a Robot Policy

LoRA fine-tune a real-robot vision-language-action model — **MolmoAct2**, trained
on a physical Franka Panda arm — onto the **LIBERO** simulator, evaluate the
checkpoint you just produced, then drive it in natural language in a live
simulator. The last notebook swaps the policy for **FastWAM**, a Wan2.2-TI2V-5B
*world*-action model, and watches it imagine the future video as well as the
next actions.

Everything runs locally on AMD ROCm. Nothing is re-hosted: the notebooks pull
pinned, immutable Hugging Face revisions of public checkpoints and datasets.

## Build & Run

```sh
ryzers build jupyterlab roscon26-finetuning
ryzers run "jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
```

Then open `0_overview.ipynb` and pick the **ROSCon26 Fine-tuning (MolmoAct2)**
kernel. All four notebooks already declare it, so opening them selects it.

JupyterLab and its `8888` port mapping come from the composed
[`jupyterlab`](../../../ide/jupyterlab) ryzer, so build the chain rather than
this package alone. Building it standalone still works — the test warns instead
of failing — but there is no notebook server to open.

```sh
ryzers run                                                   # smoke test (default CMD)
ryzers run /ryzers/demo_roscon26_finetuning.sh molmoact2     # load the DROID base, one forward
ryzers run /ryzers/demo_roscon26_finetuning.sh fastwam       # load FastWAM, one infer_action
ryzers run /ryzers/demo_roscon26_finetuning.sh sim           # interactive LIBERO at :8080
```

Run the notebooks in order:

1. `0_overview.ipynb`
2. `1_finetune_molmoact2_libero.ipynb` — download the base + dataset, open-loop
   sanity check, LoRA fine-tune, closed-loop LIBERO eval
3. `2_interactive_sim_molmoact2_libero.ipynb` — drive your checkpoint live
4. `3_inference_fastwam_libero.ipynb` — FastWAM interactive sim + video imagination

## Requirements

- A supported AMD ROCm GPU. The workshop target is Strix Halo (`gfx1151`).
- **~150 GB free disk** to run all three notebooks: roughly 35 GB of image plus
  ~90 GB of downloaded public inputs and generated checkpoints.
- A reliable connection for the first run. `HF_TOKEN` is optional but avoids
  anonymous Hub rate limits — set it in [`config.yaml`](config.yaml).

## What is in the image

| Path | Contents |
|------|----------|
| `/ryzers/notebooks/*.ipynb` | The four workshop notebooks |
| `/ryzers/notebooks/scripts/` | `ft_helpers.py` (download/stream/decode plumbing), the two interactive sim servers, the FastWAM videogen and smoke backends |
| `/opt/train-venv` | MolmoAct2 training stack — the `roscon26-finetuning` kernel |
| `/opt/fastwam-venv` | FastWAM + its own LIBERO sim stack (subprocess only) |
| `/opt/molmoact2` | `allenai/molmoact2` at a pinned commit; supplies the DROID `Policy` loader |
| `/opt/lerobot-allenai` | The `allenai/lerobot` **fork** (0.5.2) source the train-venv installs from |
| `/opt/LIBERO` + `/opt/sim/sim_libero` | The FastWAM venv's LIBERO checkout and the AMD sim harness |
| `/opt/libero-config`, `/opt/libero-config-fastwam` | One LIBERO config path per venv, plus the baked simulator meshes |

## Baked at build vs downloaded at run time

Baked (≈ a few GB on top of the base image): every pinned source checkout, both
Python environments, and the LIBERO simulator assets — the `bddl` task
definitions, `robosuite` meshes and init states that
`download_assets_from_huggingface()` fetches. Those are small, never change for
a given pin, and are what the first notebook cell would otherwise block on.

Downloaded on first use, into the host directories
[`config.yaml`](config.yaml) mounts:

| What | Size | Where | Pulled by |
|------|------|-------|-----------|
| `allenai/MolmoAct2-DROID` | ~22 GB | `$HF_HOME` | notebook 1 |
| `allenai/MolmoAct2-LIBERO-Dataset` | ~33 GB | `$HF_HOME` + `$HF_LEROBOT_HOME` | notebook 1 |
| `allenai/MolmoAct2-DROID-Dataset` episodes | ~1 GB | `$HF_HOME` | notebook 1 (3 clips per episode, `N_DROID_EPISODES=2`) |
| `allenai/MolmoAct2-LIBERO` | ~11 GB | `$HF_HOME` | notebook 2, only if `USE_PUBLIC_CHECKPOINT = True` |
| `yuanty/fastwam` + Wan2.2 T5/VAE/tokenizer | ~25 GB | `$FASTWAM_CACHE` | notebook 3 |
| `yuanty/LIBERO-fastwam` replay archive | 1.35 GB | `$FASTWAM_CACHE` | notebook 3 (`FASTWAM_DOWNLOAD_DATASET=0` skips it) |

Nothing on that list is baked, deliberately. The workshop README upstream
budgets 150 GB; baking it would produce an image no registry would accept and
would make every pin a rebuild. All downloads are idempotent and pinned to
immutable Hub commits, so a later change to a repository's `main` branch cannot
silently change the workshop.

Because `ryzers run` is `--rm`, those caches **must** be host directories or
every run re-downloads. [`config.yaml`](config.yaml) bind-mounts four:

| Host | Container | Holds |
|------|-----------|-------|
| `~/.cache/huggingface` | `/root/.cache/huggingface` | `$HF_HOME`, `$HF_LEROBOT_HOME` |
| `~/.cache/fastwam` | `/root/.cache/fastwam` | `$FASTWAM_CACHE` |
| `workspace/checkpoints` | `/ryzers/checkpoints` | `$CHECKPOINTS_DIR` — fine-tune output |
| `workspace/outputs` | `/ryzers/outputs` | `$OUT_DIR` — plots, videos, eval JSON, server logs |

The container runs as **root**, so files written into the two `~/.cache` paths
are root-owned on the host. `workspace/` sits inside the package directory,
which is also the Docker build context, so [`.dockerignore`](.dockerignore)
excludes it — otherwise every rebuild after a workshop run would upload the
~11 GB fine-tune checkpoints to the daemon.

## Why there are two virtual environments

The two stacks genuinely cannot share a `site-packages`:

| | `/opt/train-venv` | `/opt/fastwam-venv` |
|---|---|---|
| numpy | 2.x (lerobot pins `>=2.0,<2.3`) | 1.26.4 (FastWAM + the LIBERO sim era) |
| transformers | 5.4–5.5 | 4.49.0 |
| huggingface-hub | ≥0.36 | 0.29.2 |
| LIBERO | `hf-libero` wheel, config at `/opt/libero-config` | `/opt/LIBERO` checkout, config at `/opt/libero-config-fastwam` |

Both reuse the base image's ROCm torch rather than installing their own, held in
place by a `PIP_CONSTRAINT` file frozen from the base. Notebook 3 therefore runs
in the train-venv kernel and shells out to `/opt/fastwam-venv/bin/python` for
every model call — the same split the upstream course uses, and the same
reasoning behind the separate environments in
[`roscon26-rllearning`](../roscon26-rllearning) and
[`capx`](../../../robotics/capx).

### How those venvs see the base torch

This is the one place the port differs materially from upstream and is worth
knowing about if the build fails.

The Ryzers default init image (`rocm/pytorch:rocm7.14_*`) already runs inside a
virtualenv at `/opt/venv` — `VIRTUAL_ENV` and `PATH` both point there, and that
is where its ROCm torch lives. `python3 -m venv --system-site-packages` run from
*inside* a venv resolves "system" to `sys.base_prefix` (`/usr`), **not** to the
parent venv, so the flag alone would not expose the base torch. The Dockerfile
therefore records the base interpreter's `purelib` and drops it into each new
venv as a `.pth` file. A `.pth`-added directory is appended to `sys.path`, so
anything installed into the venv still shadows the base copy — the same
precedence `--system-site-packages` gives, including pip's refusal to uninstall
a distribution that lives outside the target environment. Each venv then
asserts `torch.version.hip` at build time, so a wrong guess fails the build
rather than the workshop.

## Version pins

Git checkouts, all immutable commits, all confirmed resolvable against the
GitHub API while porting:

| Repo | Commit | Why pinned there |
|------|--------|------------------|
| `allenai/molmoact2` | `cdf4b772` | The DROID `Policy` loader the local-checkpoint patch applies to |
| `allenai/lerobot` | `a4f15bf3` | The MolmoAct2 policy + LIBERO env wiring; reports `0.5.2` |
| `yuantianyuan01/FastWAM` | `45d8e145` | The two vendored patches apply to exactly this tree |
| `Lifelong-Robot-Learning/LIBERO` | `8f1084e3` | **New:** upstream cloned the default branch unpinned. Pinned to the master tip the course was validated against |

Hugging Face revisions (in `ft_helpers.py` and the notebooks) — all eight
confirmed to resolve: `allenai/MolmoAct2-DROID@d8c1abd8`,
`allenai/MolmoAct2-LIBERO-Dataset@fe3ead44`,
`allenai/MolmoAct2-DROID-Dataset@e44d3138`,
`allenai/MolmoAct2-LIBERO@0d24a92b`, `yuanty/fastwam@8eaceeb2`,
`yuanty/LIBERO-fastwam@ee018b99`, `Wan-AI/Wan2.2-TI2V-5B@921dbaf3`,
`Wan-AI/Wan2.1-T2V-1.3B@37ec5126`.

PyPI pins in the FastWAM venv, all confirmed present with a usable
distribution for CPython 3.12: `numpy==1.26.4`, `mujoco==3.3.2`,
`robosuite==1.4.0`, `bddl==1.0.1`, `gym==0.25.2`, `future==1.0.0`,
`cloudpickle==3.1.2`, `deepspeed==0.18.5`.

### The lerobot fork's torch bound had to be widened

`allenai/lerobot@a4f15bf3` declares `torch>=2.7,<2.12.0`,
`torchvision>=0.22.0,<0.27.0` and `torchcodec<0.12.0`. The Ryzers default base
is **torch 2.12.0**, which those bounds exclude — left alone, pip would pull a
CUDA torch 2.11 from PyPI over the ROCm build or fail outright against the
constraint file. The Dockerfile widens each upper bound by one minor with the
identical `sed` that [`packages/robotics/lerobot`](../../../robotics/lerobot)
applies to `huggingface/lerobot` on this same base, guarded by `grep` so an
upstream rename fails the build instead of silently not patching.

**Do not compose `packages/robotics/lerobot` with this package.** That ryzer
installs `huggingface/lerobot` **v0.6.0** into the base `site-packages`. This
course needs the `allenai` **fork** at 0.5.2, which carries the `molmoact2`
policy that upstream v0.6.0 does not, and `test.sh` asserts
`lerobot.__version__ == "0.5.2"`. The fork is installed into `/opt/train-venv`,
which shadows the base copy, so the two would not hard-conflict — but the
resulting image would have two different LeRobots on one path for no benefit.

## Changed or dropped from the upstream course

| Upstream | Here | Why |
|----------|------|-----|
| `USER jovyan`, `HOST_RENDER_GID`, `groupadd hostrender`, `chown -R jovyan` | removed | That machinery exists only because JupyterHub mounts a PVC over `/home/jovyan`. Ryzers runs as root and adds `--group-add render/video` at run time |
| Caches kept outside `/home/jovyan` under `/opt` | caches at `$HF_HOME`, `$FASTWAM_CACHE`, `$CHECKPOINTS_DIR`, `$OUT_DIR` | No PVC to hide them. They are bind-mounted instead |
| `ASSETS_ZIP` / `FINAL=with-assets` offline bundle, with BuildKit `--build-context assets=...` and the `assets-staging` / `fastwam-staging` stages | **dropped** | `ryzers/builder.py` runs a plain `docker build <package_dir>` with no `--build-context`, so the stage could never be selected. The bundle-only scripts (`fetch_assets.sh`, `unpack_bundle.sh`, `stage_assets.sh`, `prestage_*.sh`, `reconstruct_reference.py`, `make_*`) are not shipped. `ASSETS_DIR` staging inside `ft_helpers.py` is kept — it is inert when unset |
| `jupyter-server-proxy` + `JUPYTERHUB_SERVICE_PREFIX` routing for the sim UI | dropped | `RYZERS_DEFAULT_RUN_FLAGS` contains `--network=host`, so the sim's `IFrame` loads `http://localhost:8080/` directly. The notebooks already had that fallback; the markdown around it was updated |
| Train-venv registered as the `python3` kernel, overriding the default | separate `roscon26-finetuning` kernelspec, notebook metadata updated to match | Overriding `python3` would mean writing into the base venv's `share/jupyter` and hijacking JupyterLab's default kernel |
| `/repos/molmoact2` fallback path, `/outputs`, `/checkpoints`, `/home/jovyan/outputs` defaults in the notebooks | `/opt/molmoact2`, `/ryzers/outputs`, `/ryzers/checkpoints` | Match the env the image actually sets |
| LIBERO cloned from the default branch | pinned commit | Reproducibility |
| `ffmpeg` / `libav*` not installed | installed | `auplc-base` already had them; `rocm/pytorch` may not, and `torchcodec` links against them |
| `deepspeed` installed implicitly as a FastWAM dependency | installed explicitly with `--no-build-isolation` first | It is sdist-only with no `pyproject.toml`, so it compiles against whatever torch is importable. Making that explicit also documents why `setuptools` is seeded into the venv (Python 3.12 venvs no longer do) |

## Verification status

**Built and smoke-tested on the target: Ryzen AI MAX+ 395 / Radeon 8060S
(gfx1151)**, and on an RX 7900 XTX. Both venvs resolve onto the base ROCm torch
`2.12.0+rocm7.14.0` with zero `nvidia-*` wheels, `lerobot` lands at the required
`0.5.2` from the allenai fork, transformers 5.5.4, and FastWAM + LIBERO import
in the isolated `fastwam-venv` on numpy 1.26.4. `torch.cuda.is_available()` is
true and a GPU matmul runs on the 8060S (64 GiB unified memory).

The three risks flagged during the port all resolved favourably once built:
the base-site-packages `.pth` works (the base image *is* a venv at `/opt/venv`
with `include-system-site-packages = false`, so the trick was necessary);
`/opt/torch-constraints.txt` populates correctly with all three pins; and
`hf-libero` coexists with the training stack.

**Still not exercised:** no fine-tuning step, rollout or FastWAM inference has
been run — `demo.sh` does that but downloads ~25 GB on first use.

Statically checked during the port:

- All four git commit pins resolve (GitHub API).
- All eight Hugging Face revisions resolve (Hub API).
- Every PyPI pin exists with a distribution usable on CPython 3.12.
- `deepspeed==0.18.5` ships no `pyproject.toml`, so pip uses the legacy
  non-isolated build path — which is what lets it compile against ROCm torch.
- The `allenai/lerobot` pyproject really does carry the three version strings
  the `sed` rewrites, at the pinned commit.
- Notebook kernelspec metadata, path defaults and markdown edits (each applied
  with an assertion that the target string was found exactly once).

**Resolved by building — no longer open:**

- The base-site-packages `.pth`: the base image *is* a venv (`/opt/venv`,
  `include-system-site-packages = false`, `sys.base_prefix = /usr`), so
  `--system-site-packages` alone would not have exposed the base ROCm torch.
  The `.pth` is doing real work, and both venvs import torch with HIP.
- The torch constraint string: `/opt/torch-constraints.txt` resolves to
  `torch==2.12.0+rocm7.14.0`, `torchvision==0.27.0+rocm7.14.0`,
  `torchaudio==2.11.0+rocm7.14.0`. Not a pre-release, and no `nvidia-*` wheel
  reaches either venv.
- `hf-libero` + numpy 2.x: `import robosuite, mujoco` succeeds in both venvs,
  and the LIBERO metadata loads (10 tasks, 5 suites).

**Still open — look here first if something misbehaves at the workshop:**

1. **torchcodec ABI.** `<0.13.0` selects torchcodec 0.12.0, which upstream
   pairs with torch 2.12, but only its presence was confirmed, not its ABI
   match. Notebook 1 passes `--dataset.video_backend=pyav`, so this may never
   matter.
2. **EGL rendering.** `MUJOCO_GL=egl` needs a working EGL vendor ICD inside the
   container. The smoke test does not render, so the headless path is still
   unproven. `libgl1-mesa-dri` and `libosmesa6` are installed, so
   `MUJOCO_GL=osmesa` is available as a software fallback.
3. **No model actually run.** `torch.cuda.is_available()` is true and a GPU
   matmul works, but no fine-tune step, rollout or FastWAM inference has been
   executed. `demo.sh` does that; it downloads ~25 GB on first use.

## Useful overrides

- `STEPS=10000` — turn the default 10-step fine-tune smoke test into a real run
- `FT_MODE=action_expert_only|full` — freeze the VLM, or fully fine-tune
- `N_DROID_EPISODES=1` — shrink the open-loop DROID download and runtime
- `USE_PUBLIC_CHECKPOINT = True` in notebook 2 — download AllenAI's fully
  trained LIBERO release for comparison
- `POLICY_PATH=/path/to/pretrained_model` — use a compatible local LeRobot checkpoint
- `FASTWAM_DOWNLOAD_DATASET=0` — skip the FastWAM replay archive
- `SUITE`, `TASK_ID`, `N_EPISODES`, `RT_PORT` — eval and sim selection

## References

- MolmoAct2: <https://github.com/allenai/molmoact2> · <https://huggingface.co/allenai/MolmoAct2-DROID>
- LeRobot (AllenAI fork): <https://github.com/allenai/lerobot>
- LIBERO: <https://github.com/Lifelong-Robot-Learning/LIBERO>
- FastWAM: <https://github.com/yuantianyuan01/FastWAM> · <https://huggingface.co/yuanty/fastwam>
- Wan2.2-TI2V-5B: <https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B>
- LoRA: <https://arxiv.org/abs/2106.09685>

Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
