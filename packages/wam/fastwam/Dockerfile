# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Slim FastWAM (Wan2.2-TI2V-5B) policy layer for Strix Halo (gfx1151). Installs only the
# model on the base ROCm torch (CUDA pins stripped); ships no simulator. Composes on the
# plain ROCm base or on simulation/{libero,robotwin} bases via a torch+numpy constraint pin.

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

# KV-cache patch: bit-exact planning-path caching (~1.45x). Off with FASTWAM_TEXT_KV_CACHE=0.
COPY scripts/patches/fastwam_kv_cache.patch /tmp/fastwam_kv_cache.patch
RUN git -C /repos/fastwam apply --verbose /tmp/fastwam_kv_cache.patch

# Pin base torch+numpy so the install never replaces them (composes on numpy 2.x or 1.26.4).
COPY scripts/strip_cuda_torch.py /ryzers/scripts/strip_cuda_torch.py
RUN python /ryzers/scripts/strip_cuda_torch.py /repos/fastwam/pyproject.toml && \
    pip freeze | grep -iE "^(torch|torchvision|torchcodec|numpy)==" > /opt/fastwam-constraints.txt && \
    PIP_CONSTRAINT=/opt/fastwam-constraints.txt pip install --no-cache-dir -e /repos/fastwam && \
    PIP_CONSTRAINT=/opt/fastwam-constraints.txt pip install --no-cache-dir \
      "hf-transfer>=0.1.8" "sentencepiece>=0.2" "pyarrow>=17" "av>=12" \
      "imageio>=2.34" "imageio-ffmpeg>=0.5" "matplotlib>=3.8" "opencv-python-headless"

# Sim policy adapters (wrap FastWAM); imported only on a sim base via POLICY_FACTORY, unused otherwise.
COPY scripts/adapters/ /opt/fastwam-adapters/

RUN python -c "import torch; assert torch.version.hip, 'torch is not a ROCm build: '+torch.__version__; import fastwam; from fastwam.runtime import create_fastwam; print('OK fastwam on torch', torch.__version__, 'hip', torch.version.hip)"

COPY test.py /ryzers/
COPY scripts/ /ryzers/scripts/
COPY demos/ /ryzers/demos/
RUN chmod +x /ryzers/scripts/*.sh /ryzers/demos/*.sh

CMD ["python", "/ryzers/test.py"]
