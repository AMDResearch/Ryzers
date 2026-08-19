# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# FastWAM (Wan2.2-TI2V-5B world-action model) policy/model layer for AMD Strix Halo
# (gfx1151). This is a slim policy image: it installs only the FastWAM model on the
# base image's ROCm torch (upstream CUDA torch pins are stripped so the base ROCm torch
# is preserved) plus a LIBERO Policy adapter. It ships NO simulator. It composes on:
#   * the plain ROCm base  -> non-sim demos (open-loop / videogen)
#   * simulation/libero    -> closed-loop + interactive LIBERO (ryzers build libero fastwam)
#   * simulation/robotwin  -> closed-loop + interactive RoboTwin (via its own adapter)
# Installing under a constraint pinning the base's torch+numpy lets the exact same layer
# compose on a plain base (numpy 2.x) or on a sim base (numpy 1.26.4) without conflict.
# Weights, datasets and LIBERO assets are fetched at runtime / supplied by the sim base.

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG FASTWAM_COMMIT=45d8e1458921d83f8ad6cf9ce993d371208dabd0

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /ryzers

RUN apt-get update && \
    apt-get install -y --no-install-recommends git cmake build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/yuantianyuan01/FastWAM /repos/fastwam && \
    git -C /repos/fastwam checkout ${FASTWAM_COMMIT}

# Exact inference-time caches for the planning path (bit-identical output, ~1.45x faster on
# gfx1151): (1) memoize the UMT5-XXL text encode across a replan episode, (2) cache the action
# cross-attention K/V + text-embedding across a plan's diffusion steps. Shipped as a reviewable
# diff against the pinned upstream commit (upstream caches only video self-attn K/V). Default-on
# at runtime; disable with FASTWAM_TEXT_KV_CACHE=0.
COPY patches/fastwam_kv_cache.patch /tmp/fastwam_kv_cache.patch
RUN git -C /repos/fastwam apply --verbose /tmp/fastwam_kv_cache.patch

# Pin the base's torch + numpy so the FastWAM install (and its extras) never replaces
# them. This is what lets one Dockerfile compose on the plain ROCm base (numpy 2.x) and
# on a simulator base (numpy 1.26.4) identically.
COPY scripts/strip_cuda_torch.py /ryzers/scripts/strip_cuda_torch.py
RUN python /ryzers/scripts/strip_cuda_torch.py /repos/fastwam/pyproject.toml && \
    pip freeze | grep -iE "^(torch|torchvision|torchcodec|numpy)==" > /opt/fastwam-constraints.txt && \
    PIP_CONSTRAINT=/opt/fastwam-constraints.txt pip install --no-cache-dir -e /repos/fastwam && \
    PIP_CONSTRAINT=/opt/fastwam-constraints.txt pip install --no-cache-dir \
      "hf-transfer>=0.1.8" "sentencepiece>=0.2" "pyarrow>=17" "av>=12" \
      "imageio>=2.34" "imageio-ffmpeg>=0.5" "matplotlib>=3.8" "opencv-python-headless"

# Sim Policy adapters (sim_libero.Policy / sim_robotwin.Policy wrapping the FastWAM model).
# Only imported when driving a sim base via POLICY_FACTORY=fastwam_{libero,robotwin}_policy:
# build_policy; harmless (unused) on the plain base. On its own dir so demos add it to
# PYTHONPATH. The RoboTwin closed-loop instead uses the repo's own experiments/robotwin/
# fastwam_policy plugin via RoboTwin's script/eval_policy.py.
COPY adapters/ /opt/fastwam-adapters/

RUN python -c "import torch; assert torch.version.hip, 'torch is not a ROCm build: '+torch.__version__; import fastwam; from fastwam.runtime import create_fastwam; print('OK fastwam on torch', torch.__version__, 'hip', torch.version.hip)"

COPY test.py /ryzers/
COPY scripts/ /ryzers/scripts/
COPY demos/ /ryzers/demos/
RUN chmod +x /ryzers/scripts/*.sh /ryzers/demos/*.sh

CMD ["python", "/ryzers/test.py"]
