# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# MolmoAct2 (Ai2 VLA) on AMD Strix Halo (gfx1151). Adds MolmoAct2 inference plus the
# DROID and LIBERO demo stacks on the Ryzer ROCm base. The base ROCm torch is kept;
# guards fail the build if anything swaps it for a CUDA wheel.

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG MOLMOACT2_COMMIT=cdf4b7728e0545366bf3366639fd86477ef59d04
ARG LEROBOT_REF=molmoact2-hf-inference

WORKDIR /ryzers

ENV HSA_OVERRIDE_GFX_VERSION=11.5.1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

# Inference deps, minus torch/torchvision (CUDA wheels would clobber ROCm torch).
RUN pip install --no-cache-dir \
      "transformers>=4.57,<4.58" \
      "accelerate>=1.0" \
      "safetensors>=0.4" \
      "huggingface-hub[cli]>=0.36" \
      "hf-transfer>=0.1.8" \
      "pillow>=10" \
      "numpy>=1.26,<3" \
      "einops>=0.7" \
      "sentencepiece>=0.2" \
      "protobuf>=4.25" \
      "fastapi>=0.116" \
      "uvicorn[standard]>=0.35" \
      "json-numpy>=2.1.0"

# DROID demo deps: parquet, mp4 decode, action plots.
RUN pip install --no-cache-dir \
      "pyarrow>=17" \
      "av>=12" \
      "imageio[ffmpeg]>=2.34" \
      "matplotlib>=3.8"

# Upstream MolmoAct2 source, pinned.
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && \
    apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/* && \
    rm -rf /repos/molmoact2 && \
    git clone https://github.com/allenai/molmoact2 /repos/molmoact2 && \
    git -C /repos/molmoact2 checkout ${MOLMOACT2_COMMIT}

RUN python -c "import torch; assert torch.version.hip, 'torch is not a ROCm build: '+torch.__version__; print('OK torch', torch.__version__, 'hip', torch.version.hip)"

# LIBERO sim stack in an isolated venv: it needs transformers 5.3, which would clobber
# the base 4.57 used by the DROID path. The venv reuses the base ROCm torch via a .pth,
# with a torch constraints file so sim deps cannot pull CUDA wheels.
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

# venv-local OpenCV: the venv is numpy 2.x but the base cv2 is numpy 1.x, so shadow it.
RUN /opt/libero-venv/bin/pip install --no-cache-dir --ignore-installed --no-deps opencv-python

# Seed LIBERO config and bake sim assets so the eval needs no first-run prompt or fetch.
RUN { printf 'N\nN\nN\n' | /opt/libero-venv/bin/python -c "import libero.libero"; } 2>/dev/null || true; \
    /opt/libero-venv/bin/python -c "from libero.libero.utils.download_utils import download_assets_from_huggingface as d; d()" || true

# Patch lerobot eval to dump each rollout's executed trajectory for plots.
COPY scripts/apply_action_patch.py /ryzers/apply_action_patch.py
RUN /opt/libero-venv/bin/python /ryzers/apply_action_patch.py

RUN /opt/libero-venv/bin/python -c "import torch, lerobot, transformers, robosuite, mujoco; assert torch.version.hip, 'libero venv torch not ROCm: '+torch.__version__; print('LIBERO venv OK | torch', torch.__version__, '| transformers', transformers.__version__, '| lerobot', lerobot.__version__)"

# Demos and scripts flattened into /ryzers so the run commands work after a clean build.
COPY test.py /ryzers/
COPY scripts/model_smoke.py scripts/droid_openloop_demo.py scripts/libero_action_plot.py \
     scripts/interactive_server.py scripts/interactive_server_rt.py /ryzers/
COPY demos/demo_smoke.sh demos/demo_droid.sh demos/demo_libero.sh \
     demos/demo_interactive.sh demos/demo_interactive_rt.sh scripts/download.sh /ryzers/
RUN chmod +x /ryzers/*.sh

# Default `ryzers run` is the lightweight env check (no weights).
CMD ["python", "/ryzers/test.py"]
