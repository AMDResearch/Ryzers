# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /ryzers

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    rocm-dev \
    cmake \
    pybind11-dev \
    ffmpeg \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavcodec-dev \
    libavutil-dev \
    libswresample-dev \
    libswscale-dev \
    pkg-config \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip/setuptools for modern pyproject.toml support
RUN pip install --upgrade pip setuptools

RUN pip install zmq diffusers decord cmake

# pytorch3d: patch HIP namespace for ROCm compatibility
ENV FORCE_CUDA=1
RUN git clone --depth 1 -b v0.7.9 https://github.com/facebookresearch/pytorch3d.git /ryzers/pytorch3d && \
    cd /ryzers/pytorch3d && \
    grep -rl --include="*.hip" --include="*.cuh" --include="*.h" --include="*.cu" \
        'cub::' . | xargs -r sed -i 's/cub::/hipcub::/g' && \
    grep -rl --include="*.hip" --include="*.cuh" --include="*.h" --include="*.cu" \
        '#include <cub/cub.cuh>' . | xargs -r sed -i 's|#include <cub/cub.cuh>|#include <hipcub/hipcub.hpp>|g' && \
    pip install --no-build-isolation -e .

# ROCm flash-attention (triton backend)
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
ENV BUILD_TYPE="rocm"
RUN git clone --recursive https://github.com/ROCm/flash-attention && \
    cd flash-attention && \
    git checkout main_perf && \
    pip install .

# torchcodec: strip -Werror for ROCm build
ENV PYTORCH_ROCM_ARCH="gfx1151"
RUN git clone --depth 1 -b v0.7.0 https://github.com/pytorch/torchcodec.git && \
    cd /ryzers/torchcodec && \
    sed -i 's/-Werror//g' src/torchcodec/_core/CMakeLists.txt && \
    pip install --no-build-isolation -e .

RUN git clone https://github.com/NVIDIA/Isaac-GR00T && \
    cd Isaac-GR00T && \
    git checkout 4af2b62 && \
    pip install -e .

COPY test.py /ryzers/test.py
COPY test.sh /ryzers/test.sh
RUN chmod +x /ryzers/test.sh

CMD ["/bin/bash", "-c", "/ryzers/test.sh"]
