# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-lc"]

# ---------------------------------------------------------------------------
# Upstream pins. Every one of these was resolved against its host (GitHub API /
# PyPI JSON) while porting; see README.md for the verification table. All four
# git pins are immutable commits, not branches, so a later upstream push cannot
# silently change the workshop.
# ---------------------------------------------------------------------------
ARG MOLMOACT2_COMMIT=cdf4b7728e0545366bf3366639fd86477ef59d04
ARG LEROBOT_COMMIT=a4f15bf347dee7eb8a8c5f4a70a37f476b091113
ARG FASTWAM_COMMIT=45d8e1458921d83f8ad6cf9ce993d371208dabd0
# Upstream cloned LIBERO from the default branch. Pinned here to the master tip
# the course was validated against so the sim stack cannot drift mid-workshop.
ARG LIBERO_COMMIT=8f1084e3132a39270c3a13ebe37270a43ece2a01

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_XET=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    TOKENIZERS_PARALLELISM=false \
    TRAIN_VENV=/opt/train-venv \
    FASTWAM_VENV=/opt/fastwam-venv

WORKDIR /ryzers

# Build toolchain, EGL/OSMesa for headless MuJoCo rendering, and the FFmpeg
# shared libraries torchcodec links against (the `lerobot` ryzer installs the
# same set for the same reason).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates pkg-config python3-dev python3-venv \
        libegl1 libegl-dev libgl1 libgles2 libgles-dev libglu1-mesa \
        libgl1-mesa-dri libglew-dev libglib2.0-0t64 \
        libosmesa6 libosmesa6-dev \
        ffmpeg libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
        libswscale-dev libswresample-dev libavfilter-dev \
        fonts-dejavu-core \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Base interpreter discovery.
#
# The Ryzers default init image (rocm/pytorch:rocm7.14_*) already runs inside a
# virtualenv at /opt/venv -- PATH and VIRTUAL_ENV both point there, and that is
# where its ROCm torch lives. `python3 -m venv --system-site-packages` from
# *inside* a venv resolves "system" to sys.base_prefix (i.e. /usr), NOT to the
# parent venv, so the flag alone would NOT expose the base torch. Recording the
# base interpreter's purelib here and dropping it into each new venv as a .pth
# works whether or not the base interpreter is itself a venv.
#
# A .pth-added directory is *appended* to sys.path, so anything installed into
# the venv still shadows the base copy -- which is exactly the --system-site-
# packages semantics the upstream course relied on, including pip's refusal to
# uninstall a distribution that lives outside the target environment.
# ---------------------------------------------------------------------------
RUN python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' \
      > /opt/base-site-packages.txt \
    && echo "base site-packages: $(cat /opt/base-site-packages.txt)" \
    && python3 -c "import torch; assert torch.version.hip, torch.__version__; \
print('base torch', torch.__version__, 'hip', torch.version.hip)"

# ---------------------------------------------------------------------------
# Source checkouts.
#
# molmoact2 supplies the DROID `Policy` loader notebook 1 imports; the allenai
# lerobot FORK (not huggingface/lerobot) supplies the `molmoact2` policy and the
# LIBERO env wiring, and it reports version 0.5.2. Ryzers' own
# packages/robotics/lerobot installs huggingface/lerobot v0.6.0 into the BASE
# site-packages, which would shadow-conflict with this fork -- do not compose
# the two (see README.md).
# ---------------------------------------------------------------------------
RUN git clone https://github.com/allenai/molmoact2 /opt/molmoact2 \
    && git -C /opt/molmoact2 checkout "${MOLMOACT2_COMMIT}" \
    && mkdir -p /opt/lerobot-allenai \
    && git -C /opt/lerobot-allenai init -q \
    && git -C /opt/lerobot-allenai remote add origin https://github.com/allenai/lerobot.git \
    && git -C /opt/lerobot-allenai fetch --depth 1 -q origin "${LEROBOT_COMMIT}" \
    && git -C /opt/lerobot-allenai checkout -q --detach FETCH_HEAD

# The fork pins "torch>=2.7,<2.12.0" / "torchvision>=0.22.0,<0.27.0" and
# "torchcodec<0.12.0", all of which EXCLUDE the Ryzers default base image
# (torch 2.12.0). Left alone, pip would try to pull a CUDA torch 2.11 from PyPI
# over the ROCm build, or fail outright against the torch constraint below.
# Widening the upper bounds by one minor is the identical fix
# packages/robotics/lerobot applies to huggingface/lerobot on this same base.
# The grep guards make a silent upstream rename fail the build instead of
# producing a quietly unpatched pyproject.
RUN cd /opt/lerobot-allenai \
    && grep -q '"torch>=2.7,<2.12.0"' pyproject.toml \
    && grep -q '"torchvision>=0.22.0,<0.27.0"' pyproject.toml \
    && grep -q '<0.12.0' pyproject.toml \
    && sed -i \
        -e 's/"torch>=2.7,<2.12.0"/"torch>=2.7,<2.13.0"/' \
        -e 's/"torchvision>=0.22.0,<0.27.0"/"torchvision>=0.22.0,<0.28.0"/' \
        -e 's/<0.12.0/<0.13.0/g' \
        pyproject.toml \
    && grep -nE '"(torch|torchvision|torchcodec)[>=<]' pyproject.toml

# ===========================================================================
# 1. MolmoAct2 training environment (/opt/train-venv)
#
# Backs notebooks 0-2 and the Hugging Face plumbing of notebook 3. numpy 2.x,
# transformers 5.4-5.5, lerobot 0.5.2. Isolated from the base interpreter so a
# composed ryzer's torch stack is never rewritten, and from the FastWAM venv
# below, which needs numpy 1.26.4 and transformers 4.49.
#
# /opt/train-venv is NOT a free choice of path: notebooks 1 and 2 invoke
# /opt/train-venv/bin/python and /opt/train-venv/bin/lerobot-eval by absolute
# path, so renaming it means editing the notebooks.
# ===========================================================================
RUN python3 -m venv --system-site-packages ${TRAIN_VENV} \
    && VENV_SP="$(${TRAIN_VENV}/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" \
    && printf '%s\n' "$(cat /opt/base-site-packages.txt)" > "${VENV_SP}/zz_base_site_packages.pth" \
    && ${TRAIN_VENV}/bin/pip install --no-cache-dir --upgrade pip wheel "setuptools<81" \
    && ${TRAIN_VENV}/bin/python -c "import torch; assert torch.version.hip, torch.__version__; \
print('train-venv torch', torch.__version__)"

# Freeze the base image's torch stack so nothing in the dependency tree can
# replace the ROCm build with a PyPI (CUDA) wheel. torchaudio is optional: some
# rocm/pytorch tags ship it, some do not.
RUN for pkg in torch torchvision torchaudio; do \
        v="$(${TRAIN_VENV}/bin/python -c "from importlib.metadata import version; print(version('${pkg}'))" 2>/dev/null)" \
          && echo "${pkg}==${v}" || true; \
    done > /opt/torch-constraints.txt \
    && echo "--- /opt/torch-constraints.txt ---" && cat /opt/torch-constraints.txt

# `setuptools<81` is deliberate: robosuite 1.4.0 / robomimic 0.2.0 (pulled in by
# lerobot[libero] -> hf-libero) still import pkg_resources, removed in 81.
RUN PIP_CONSTRAINT=/opt/torch-constraints.txt ${TRAIN_VENV}/bin/pip install --no-cache-dir \
      "/opt/lerobot-allenai[libero,molmoact2,training]" \
      "accelerate>=1.0" \
      "av>=12" \
      "einops>=0.7" \
      "fastapi>=0.116" \
      "hf-transfer>=0.1.8" \
      "huggingface-hub[cli]>=0.36" \
      "imageio[ffmpeg]>=2.34" \
      "ipykernel" \
      "json-numpy>=2.1.0" \
      "matplotlib>=3.8" \
      "pyarrow>=17" \
      "safetensors>=0.4" \
      "sentencepiece>=0.2" \
      "uvicorn[standard]>=0.35" \
    && ${TRAIN_VENV}/bin/pip install --no-cache-dir --ignore-installed --no-deps opencv-python \
    && ${TRAIN_VENV}/bin/python -c \
      "import torch, lerobot, transformers, robosuite, mujoco; \
assert torch.version.hip, torch.__version__; \
assert lerobot.__version__ == '0.5.2', lerobot.__version__; \
print('MolmoAct2 training environment OK:', torch.__version__, transformers.__version__, lerobot.__version__)" \
    && rm -f /tmp/robosuite.log

# Seed LIBERO non-interactively (its first import otherwise prompts on stdin)
# and bake the robosuite/bddl simulator meshes into the image. The config lives
# at /opt/libero-config rather than under $HOME so it cannot be shadowed by a
# bind mount at run time.
RUN mkdir -p /opt/libero-config \
    && printf 'N\n' | LIBERO_CONFIG_PATH=/opt/libero-config \
      ${TRAIN_VENV}/bin/python -c "import libero.libero" \
    && LIBERO_CONFIG_PATH=/opt/libero-config \
      ${TRAIN_VENV}/bin/python -c \
      "from libero.libero import get_libero_path; \
from libero.libero.utils.download_utils import download_assets_from_huggingface as download; \
download(get_libero_path('assets'))" \
    && rm -f /tmp/robosuite.log

# ===========================================================================
# 2. FastWAM world-action model (/opt/fastwam-venv) -- powers notebook 3.
#
# FastWAM pins numpy==1.26.4, transformers==4.49.0 and huggingface-hub==0.29.2;
# the train-venv above is on numpy 2.x and transformers 5.4+. The two cannot
# share a site-packages, hence a second isolated venv with its OWN LIBERO
# checkout and its OWN LIBERO config path. Notebook 3 runs in the train-venv
# kernel and shells out to ${FASTWAM_VENV}/bin/python for every model call.
#
# Placed after the train-venv so edits to the notebooks never invalidate it.
# ===========================================================================
RUN git clone -q https://github.com/yuantianyuan01/FastWAM /repos/fastwam \
    && git -C /repos/fastwam checkout -q "${FASTWAM_COMMIT}" \
    && mkdir -p /opt/LIBERO \
    && git -C /opt/LIBERO init -q \
    && git -C /opt/LIBERO remote add origin https://github.com/Lifelong-Robot-Learning/LIBERO.git \
    && git -C /opt/LIBERO fetch --depth 1 -q origin "${LIBERO_COMMIT}" \
    && git -C /opt/LIBERO checkout -q --detach FETCH_HEAD

# Vendored AMD glue: the sim_libero harness, the FastWAM LIBERO policy adapter,
# the two upstream patches and the CUDA-pin stripper. Source only, no binaries,
# so the build is self-contained from this package directory alone.
COPY fastwam/ /opt/fastwam-src/

# Two reviewable diffs against the pinned upstream:
#  - fastwam_kv_cache.patch memoizes the UMT5-XXL text encode across a replan
#    episode and caches the action cross-attention K/V. Bit-identical output,
#    ~1.45x faster on gfx1151. Default-on; FASTWAM_TEXT_KV_CACHE=0 restores
#    upstream behaviour for parity checks.
#  - fastwam_public_wan_assets.patch selects the original public Wan T5/VAE
#    files on Hugging Face instead of the converted ModelScope-only redirect.
RUN git -C /repos/fastwam apply --verbose /opt/fastwam-src/patches/fastwam_kv_cache.patch \
    && git -C /repos/fastwam apply --verbose /opt/fastwam-src/patches/fastwam_public_wan_assets.patch

# numpy 1.26.4 goes in FIRST, then the constraint file is frozen from the result
# so neither FastWAM nor LIBERO can drag numpy forward or swap the ROCm torch.
# strip_cuda_torch.py removes FastWAM's torch/torchvision/torchcodec/numpy pins
# (upstream hard-pins torch==2.7.1+cu128) before the editable install.
#
# `pip freeze` in this venv also reports the base image's distributions, which
# is what puts the ROCm torch into the constraint file.
RUN python3 -m venv --system-site-packages ${FASTWAM_VENV} \
    && VENV_SP="$(${FASTWAM_VENV}/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" \
    && printf '%s\n' "$(cat /opt/base-site-packages.txt)" > "${VENV_SP}/zz_base_site_packages.pth" \
    && ${FASTWAM_VENV}/bin/pip install --no-cache-dir --upgrade pip wheel "setuptools<81" \
    && ${FASTWAM_VENV}/bin/pip install --no-cache-dir "numpy==1.26.4" \
    && ${FASTWAM_VENV}/bin/python /opt/fastwam-src/strip_cuda_torch.py /repos/fastwam/pyproject.toml \
    && ${FASTWAM_VENV}/bin/pip freeze | grep -iE "^(torch|torchvision|torchcodec|numpy)==" \
         > /opt/fastwam-constraints.txt \
    && cat /opt/fastwam-constraints.txt

# deepspeed 0.18.5 is a hard FastWAM dependency but is sdist-only and ships no
# pyproject.toml, so pip builds it through the legacy setup.py path with build
# isolation disabled -- it compiles against whatever torch is importable. That
# is the ROCm torch here, which is what we want, but it is also why setuptools
# had to be installed into this venv first (Python 3.12 venvs no longer seed
# it). FastWAM's inference path never imports deepspeed; it is only reachable
# through accelerate during training.
#
# --no-build-isolation also means pip installs no build requirements for it, so
# the two its setup.py imports (torch, packaging) plus ninja must already be
# present.
RUN PIP_CONSTRAINT=/opt/fastwam-constraints.txt ${FASTWAM_VENV}/bin/pip install \
        --no-cache-dir "packaging" "ninja" \
    && PIP_CONSTRAINT=/opt/fastwam-constraints.txt ${FASTWAM_VENV}/bin/pip install \
        --no-cache-dir --no-build-isolation "deepspeed==0.18.5"

# LIBERO sim stack (numpy 1.26.4 era) + FastWAM itself.
RUN PIP_CONSTRAINT=/opt/fastwam-constraints.txt ${FASTWAM_VENV}/bin/pip install --no-cache-dir \
        "robosuite==1.4.0" "bddl==1.0.1" "mujoco==3.3.2" easydict thop \
        "future==1.0.0" "cloudpickle==3.1.2" "gym==0.25.2" termcolor \
        "imageio>=2.34" "imageio-ffmpeg>=0.5" "pillow>=10" \
    && PIP_CONSTRAINT=/opt/fastwam-constraints.txt ${FASTWAM_VENV}/bin/pip install \
        --no-cache-dir --no-deps -e /opt/LIBERO \
    && PIP_CONSTRAINT=/opt/fastwam-constraints.txt ${FASTWAM_VENV}/bin/pip install \
        --no-cache-dir -e /repos/fastwam \
    && PIP_CONSTRAINT=/opt/fastwam-constraints.txt ${FASTWAM_VENV}/bin/pip install --no-cache-dir \
        "hf-transfer>=0.1.8" "sentencepiece>=0.2" "pyarrow>=17" "av>=12" "matplotlib>=3.8" \
    && ${FASTWAM_VENV}/bin/pip install --no-cache-dir --ignore-installed --no-deps \
        opencv-python-headless

# Put the sim harness and the policy adapter where the demos expect them and
# register their import paths in the FastWAM venv ONLY. sim_libero_compat.pth
# installs a torch.load compatibility shim; fastwam_paths.pth adds the LIBERO
# checkout, the upstream repo, its experiments/ tree (for experiments.libero)
# and the adapter directory.
RUN mkdir -p /opt/sim /opt/fastwam-adapters \
    && cp -r /opt/fastwam-src/sim_libero /opt/sim/sim_libero \
    && cp /opt/fastwam-src/adapters/fastwam_libero_policy.py /opt/fastwam-adapters/ \
    && SP="$(${FASTWAM_VENV}/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" \
    && cp /opt/fastwam-src/sim_libero_compat.pth "$SP/" \
    && printf '/opt/LIBERO\n/repos/fastwam\n/repos/fastwam/experiments/libero\n/opt/fastwam-adapters\n' \
         > "$SP/fastwam_paths.pth"

# Seed this venv's LIBERO on its OWN config path so it can never clash with the
# train-venv's copy, and prove the whole FastWAM stack imports on the base ROCm
# torch. No GPU is available during `docker build`, so this is imports only.
RUN mkdir -p /opt/libero-config-fastwam \
    && printf 'N\n' | LIBERO_CONFIG_PATH=/opt/libero-config-fastwam \
       ${FASTWAM_VENV}/bin/python -c \
       "import mujoco, robosuite; from libero.libero import benchmark; \
from libero.libero.envs import OffScreenRenderEnv; import sim_libero; \
from sim_libero.policy import load_policy; print('sim_libero + LIBERO OK, mujoco', mujoco.__version__)" \
    && LIBERO_CONFIG_PATH=/opt/libero-config-fastwam ${FASTWAM_VENV}/bin/python -c \
       "import torch; assert torch.version.hip, 'torch is not a ROCm build: ' + torch.__version__; \
import fastwam; from fastwam.runtime import create_fastwam; \
print('FastWAM OK on torch', torch.__version__, 'hip', torch.version.hip)" \
    && rm -f /tmp/robosuite.log

# ===========================================================================
# 3. Course content
# ===========================================================================

# Let notebooks hand the local snapshot of a revision-pinned Hub download to the
# upstream DROID policy loader; loading by repo id still works for other callers.
COPY molmoact2/ /opt/molmoact2-src/
RUN git -C /opt/molmoact2 apply --verbose \
      /opt/molmoact2-src/patches/molmoact2_local_checkpoint.patch

# The workshop kernel. Written by hand rather than via `ipykernel install`
# because the composed `jupyterlab` ryzer runs out of the base image's venv:
# jupyter_core searches sys.prefix/share/jupyter BEFORE the system directories,
# so a kernelspec installed with --prefix=/usr/local only resolves when its name
# is unique, which is exactly what this one is. All four notebooks carry
# "kernelspec": {"name": "roscon26-finetuning"} so they select it on open.
RUN mkdir -p /usr/local/share/jupyter/kernels/roscon26-finetuning \
    && printf '%s\n' \
      '{' \
      '  "argv": ["/opt/train-venv/bin/python", "-m", "ipykernel_launcher", "-f", "{connection_file}"],' \
      '  "display_name": "ROSCon26 Fine-tuning (MolmoAct2)",' \
      '  "language": "python",' \
      '  "env": {' \
      '    "TORCH_BLAS_PREFER_HIPBLASLT": "0",' \
      '    "MUJOCO_GL": "egl",' \
      '    "PYOPENGL_PLATFORM": "egl"' \
      '  }' \
      '}' \
      > /usr/local/share/jupyter/kernels/roscon26-finetuning/kernel.json

COPY notebooks /ryzers/notebooks/

# Cache and artefact roots. These are bind-mounted from the host by config.yaml
# because `ryzers run` is `--rm`: without the mounts every run would re-download
# the ~60 GB of public model and dataset files the notebooks fetch.
RUN mkdir -p /root/.cache/huggingface /root/.cache/fastwam \
      /ryzers/checkpoints /ryzers/outputs

ENV HF_HOME=/root/.cache/huggingface \
    HF_LEROBOT_HOME=/root/.cache/huggingface/lerobot \
    HF_HUB_OFFLINE=0 \
    TRANSFORMERS_OFFLINE=0 \
    HF_DATASETS_OFFLINE=0 \
    LIBERO_CONFIG_PATH=/opt/libero-config \
    DROID_SERVER_DIR=/opt/molmoact2/examples/droid \
    OUT_DIR=/ryzers/outputs \
    CHECKPOINTS_DIR=/ryzers/checkpoints \
    TORCH_BLAS_PREFER_HIPBLASLT=0 \
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl

# Notebook 3 runtime knobs. The FastWAM venv uses its OWN LIBERO config path,
# never the train-venv's. DIFFSYNTH_DOWNLOAD_SOURCE selects Hugging Face over
# upstream's ModelScope default (the same public files are on both).
ENV FASTWAM_REPO=/repos/fastwam \
    LIBERO_CONFIG_PATH_FASTWAM=/opt/libero-config-fastwam \
    FASTWAM_CACHE=/root/.cache/fastwam \
    DIFFSYNTH_MODEL_BASE_PATH=/root/.cache/fastwam/diffsynth \
    DIFFSYNTH_DOWNLOAD_SOURCE=huggingface \
    FASTWAM_REDIRECT_COMMON_FILES=false \
    FASTWAM_RELEASE_DIR=/root/.cache/fastwam/fastwam_release \
    FASTWAM_DATA_DIR=/root/.cache/fastwam/data \
    FASTWAM_TEXT_KV_CACHE=1

COPY test.sh /ryzers/test_roscon26_finetuning.sh
COPY demo.sh /ryzers/demo_roscon26_finetuning.sh
RUN chmod +x /ryzers/test_roscon26_finetuning.sh /ryzers/demo_roscon26_finetuning.sh

WORKDIR /ryzers/notebooks

# Notebooks 2 and 3 serve their interactive LIBERO UI here. Ryzers runs with
# --network=host, so this is reachable at http://localhost:8080/ with no port
# mapping; the EXPOSE is documentation for anyone running the image by hand.
EXPOSE 8080

# Default command runs the test (for interactive use: jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root)
CMD ["/ryzers/test_roscon26_finetuning.sh"]
