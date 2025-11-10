# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}  

ENV DEBIAN_FRONTEND=noninteractive
ENV HSA_OVERRIDE_GFX_VERSION=11.0.0

WORKDIR /ryzers

# Basic environment setup
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HSA_OVERRIDE_GFX_VERSION=11.0.0

# System packages: Vulkan + FFmpeg + minimal GUI dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget git ca-certificates locales \
    libvulkan1 mesa-vulkan-drivers vulkan-tools \
    ffmpeg \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# pip configuration (system-wide inside the container)
RUN mkdir -p /etc/pip.conf.d/ && \
    echo "[global]" > /etc/pip.conf.d/pip.conf && \
    echo "break-system-packages = true" >> /etc/pip.conf.d/pip.conf

# Locale setup (optional but kept for completeness)
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && locale-gen

# Compatible build: NumPy 2.x with corresponding scikit-image version
RUN pip3 install --no-cache-dir --break-system-packages "numpy==2.1.2" && \
    pip3 install --no-cache-dir --break-system-packages "scikit-image==0.24.0"

# Genesis and gstaichi (compatible with Genesis 0.3.3; beta release fixes kernel interface mismatch)
RUN pip3 install --no-cache-dir --break-system-packages "gstaichi==2.5.0b4" \
 && pip3 install --no-cache-dir --break-system-packages "genesis-world==0.3.3"

# Copy test script
COPY test.py /ryzers/test_genesis.py
COPY demo.sh /ryzers/demo_genesis.sh
RUN chmod +x /ryzers/test_genesis.py /ryzers/demo_genesis.sh

RUN git clone https://github.com/Genesis-Embodied-AI/Genesis

CMD /ryzers/test_genesis.py 
