# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# MolmoAct2 (Ai2 VLA) on AMD Strix Halo (gfx1151) Ryzer package.
# Layers the MolmoAct2 inference + DROID open-loop + LIBERO closed-loop stacks on
# the default Ryzer ROCm base (rocm/pytorch, torch 2.10.0+rocm7.2.2).
#
# Three self-contained demos (see README), each run via `ryzers run`:
#   1. /ryzers/demo_smoke.sh   full-model load + one prediction (capability 1)
#   2. /ryzers/demo_droid.sh   random DROID episode, open-loop, video + GT-vs-pred plot
#   3. /ryzers/demo_libero.sh  random LIBERO scene, closed-loop MuJoCo, video + action plot
#
# We keep the base image's ROCm torch/torchvision (upstream's CUDA-only pins are not
# installed); a build-time guard fails if torch is swapped off ROCm. Other deps track
# allenai/molmoact2 @ cdf4b772.

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG MOLMOACT2_COMMIT=cdf4b7728e0545366bf3366639fd86477ef59d04
ARG LEROBOT_REF=molmoact2-hf-inference

WORKDIR /ryzers

ENV HF_HUB_ENABLE_HF_TRANSFER=0 \
    HF_HUB_DISABLE_XET=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

# MolmoAct2 server/inference deps EXCEPT torch/torchvision (cu121 wheels would
# clobber the ROCm torch already in the base image).
RUN pip install --no-cache-dir \
      "transformers>=4.57,<4.58" \
      "accelerate>=1.0" \
      "safetensors>=0.4" \
      "huggingface-hub[cli]>=0.36,<1.0" \
      "hf-transfer>=0.1.8" \
      "pillow>=10" \
      "numpy>=1.26,<3" \
      "einops>=0.7" \
      "sentencepiece>=0.2" \
      "protobuf>=4.25" \
      "fastapi>=0.116" \
      "uvicorn[standard]>=0.35" \
      "json-numpy>=2.1.0"

# Open-loop eval deps: read LeRobot v3.0 parquet (pyarrow), decode/encode the DROID
# camera mp4s (PyAV + imageio-ffmpeg; decord fails on these streams), and plot
# GT-vs-pred overlays (matplotlib). No torch coupling, so ROCm torch is untouched.
RUN pip install --no-cache-dir \
      "pyarrow>=17" \
      "av>=12" \
      "imageio[ffmpeg]>=2.34" \
      "matplotlib>=3.8"

# Bake the upstream MolmoAct2 source at the pinned commit (host_server_droid.Policy
# is the bf16 loader the DROID demos use). `rm -rf` first keeps the build idempotent
# if the base already carries a /repos/molmoact2.
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && \
    apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/* && \
    rm -rf /repos/molmoact2 && \
    git clone https://github.com/allenai/molmoact2 /repos/molmoact2 && \
    git -C /repos/molmoact2 checkout ${MOLMOACT2_COMMIT}

# Build-time guard: fail the build if anything swapped torch off ROCm.
RUN python -c "import torch; assert torch.version.hip, 'torch is not a ROCm build: '+torch.__version__; print('OK torch', torch.__version__, 'hip', torch.version.hip)"

# ============================================================================
# LIBERO simulation-eval stack (closed-loop success-rate reproduction of the
# paper's metric). Installed in an ISOLATED venv (/opt/libero-venv) on purpose:
# the allenai/lerobot fork + the allenai/MolmoAct2-*-LIBERO checkpoints require
# transformers 5.3, which would otherwise clobber the base /opt/venv transformers
# 4.57 used by the DROID inference path. The isolated venv re-uses the base ROCm
# torch/torchvision via a .pth, and a pip constraints file pins them so the sim deps
# can never pull the default CUDA torch + nvidia-cu12 wheels.
#   * MuJoCo renders headless via EGL on gfx1151 (base already ships the Mesa EGL
#     stack); we only add the -dev headers + cmake/compiler that robosuite's
#     egl_probe builds from source.
#   * fork @ branch ${LEROBOT_REF}.
# ============================================================================
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && \
    apt-get install -y --no-install-recommends \
      cmake build-essential libegl-dev libgles-dev libosmesa6-dev libglew-dev && \
    rm -rf /var/lib/apt/lists/*

RUN rm -rf /opt/lerobot-allenai && git clone --branch ${LEROBOT_REF} --depth 1 \
      https://github.com/allenai/lerobot.git /opt/lerobot-allenai

RUN rm -rf /opt/libero-venv && python -m venv /opt/libero-venv && \
    echo "/opt/venv/lib/python3.12/site-packages" \
      > "/opt/libero-venv/lib/python3.12/site-packages/zz_base_site.pth" && \
    /opt/venv/bin/pip freeze | grep -iE '^(torch|torchvision)==' > /opt/torch-constraints.txt && \
    echo "torch constraints:" && cat /opt/torch-constraints.txt && \
    PIP_CONSTRAINT=/opt/torch-constraints.txt /opt/libero-venv/bin/pip install --no-cache-dir \
      "/opt/lerobot-allenai[libero]"

# Force a venv-local OpenCV: the LIBERO venv resolves numpy 2.x, but the shared
# base /opt/venv ships a numpy-1.x-built cv2. Without this, pip sees the base cv2
# via the .pth and skips installing one into the venv, so robosuite imports the
# base cv2 and dies with "numpy.core.multiarray failed to import". --ignore-installed
# forces a numpy-2-compatible cv2 into the venv (it then shadows the base one).
RUN /opt/libero-venv/bin/pip install --no-cache-dir --ignore-installed --no-deps opencv-python

# Seed LIBERO config non-interactively + bake the sim assets into the image so the
# eval is fully self-contained (no first-run Y/N prompt, no runtime asset fetch).
RUN { printf 'N\nN\nN\n' | /opt/libero-venv/bin/python -c "import libero.libero"; } 2>/dev/null || true; \
    /opt/libero-venv/bin/python -c "from libero.libero.utils.download_utils import download_assets_from_huggingface as d; d()" || true

# Bake the AMD action-dump patch into the installed lerobot eval so every LIBERO
# rollout writes its executed 7-DoF trajectory (.npy) next to the rendered video.
COPY scripts/apply_action_patch.py /ryzers/apply_action_patch.py
RUN /opt/libero-venv/bin/python /ryzers/apply_action_patch.py

# Guard: the LIBERO venv must still resolve the ROCm torch (not a CUDA build).
RUN /opt/libero-venv/bin/python -c "import torch, lerobot, transformers, robosuite, mujoco; assert torch.version.hip, 'libero venv torch not ROCm: '+torch.__version__; print('LIBERO venv OK | torch', torch.__version__, '| transformers', transformers.__version__, '| lerobot', lerobot.__version__)"

# Demo entrypoints, embodiment assets, and env sign-of-life.
COPY assets/xarm6 /ryzers/assets/xarm6
RUN python /ryzers/assets/xarm6/fetch_meshes.py
COPY test.py /ryzers/
COPY scripts/model_smoke.py scripts/droid_openloop_demo.py scripts/libero_action_plot.py \
     scripts/interactive_server.py scripts/interactive_server_rt.py scripts/embodiment.py /ryzers/
COPY demos/demo_smoke.sh demos/demo_droid.sh demos/demo_libero.sh demos/demo_interactive.sh \
     demos/demo_interactive_rt.sh demos/demo_interactive_embodiment.sh /ryzers/
COPY scripts/_hf_common.sh scripts/download.sh scripts/download_smoke.sh scripts/download_droid.sh \
     scripts/download_libero.sh scripts/download_interactive.sh scripts/download_interactive_rt.sh \
     scripts/download_interactive_embodiment.sh /ryzers/
RUN chmod +x /ryzers/*.sh

# Default `ryzers run` (no CMD override) = lightweight env sign-of-life (no weights).
CMD ["python", "/ryzers/test.py"]
