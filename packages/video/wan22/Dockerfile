# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG WAN22_REPO=https://github.com/Wan-Video/Wan2.2.git
ARG WAN22_COMMIT=42bf4cfaa384bc21833865abc2f9e6c0e67233dc

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    python3-pip \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages \
    "opencv-python>=4.9.0.80" \
    "diffusers>=0.31.0" \
    "huggingface_hub>=0.30.0" \
    "transformers>=4.49.0,<=4.51.3" \
    "tokenizers>=0.20.3" \
    "accelerate>=1.1.1" \
    "peft>=0.13.0" \
    "tqdm" \
    "imageio[ffmpeg]" \
    "easydict" \
    "ftfy" \
    "imageio-ffmpeg" \
    "numpy>=1.23.5,<2" \
    "einops" \
    "librosa" \
    "decord"

RUN git clone ${WAN22_REPO} /opt/Wan2.2 \
 && cd /opt/Wan2.2 \
 && git checkout ${WAN22_COMMIT}

COPY flash_attn_shim /opt/wan22-overlay/python

ENV PYTHONPATH=/opt/wan22-overlay/python:/opt/Wan2.2

WORKDIR /ryzers
COPY test.sh /ryzers/test_wan22.sh
COPY download.sh /ryzers/download_wan22.sh
COPY download_wan22.py /ryzers/download_wan22.py
COPY demo.sh /ryzers/demo_wan22.sh
COPY generate_ti2v_strix.py /ryzers/generate_ti2v_strix.py
RUN chmod +x /ryzers/test_wan22.sh /ryzers/download_wan22.sh /ryzers/download_wan22.py /ryzers/demo_wan22.sh /ryzers/generate_ti2v_strix.py

CMD /ryzers/test_wan22.sh

