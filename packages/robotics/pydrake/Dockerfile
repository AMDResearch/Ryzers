# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# Strix Point gfx1150 requires HSA override for ROCm compatibility
ENV HSA_OVERRIDE_GFX_VERSION=11.0.0

WORKDIR /ryzers

# System dependencies for Drake (visualization, solvers, rendering)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git wget curl ca-certificates \
    libgl1-mesa-dev libglu1-mesa-dev libglew-dev \
    libx11-dev libxext-dev libxrender-dev \
    libegl1-mesa-dev libgles2-mesa-dev \
    libvulkan1 mesa-vulkan-drivers \
    libtbb-dev liblapack-dev libblas-dev \
    libpng-dev libjpeg-dev libtiff-dev \
    python3-dev python3-pip \
 && rm -rf /var/lib/apt/lists/*

# Install Drake via pip (stable release)
RUN pip3 install --no-cache-dir --break-system-packages \
    "drake>=1.30.0"

# Additional scientific computing packages (PyTorch ROCm comes from base image)
RUN pip3 install --no-cache-dir --break-system-packages \
    "numpy>=1.26.0" \
    "scipy>=1.12.0" \
    "matplotlib>=3.8.0" \
    "meshcat>=0.3.2" \
    "pyvirtualdisplay>=3.0" \
    "opencv-python-headless>=4.8.0"

# Copy test and demo scripts
COPY test.py /ryzers/test_pydrake.py
COPY demo.py /ryzers/demo_pydrake.py
RUN chmod +x /ryzers/test_pydrake.py /ryzers/demo_pydrake.py

# Meshcat visualization port
EXPOSE 7000

CMD /ryzers/test_pydrake.py
