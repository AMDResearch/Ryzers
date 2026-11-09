# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}  

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /ryzers

# Basic environment setup
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# System packages: FFmpeg + EGL for headless rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget git ca-certificates locales \
    ffmpeg \
    libgl1 libglib2.0-0 \
    libegl1-mesa-dev libgles2-mesa-dev \
    python3-tk \
 && rm -rf /var/lib/apt/lists/*

# pip configuration (system-wide inside the container)
RUN mkdir -p /etc/pip.conf.d/ && \
    echo "[global]" > /etc/pip.conf.d/pip.conf && \
    echo "break-system-packages = true" >> /etc/pip.conf.d/pip.conf

# Locale setup (optional but kept for completeness)
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && locale-gen

# Compatible build: NumPy 2.x with corresponding scikit-image version
RUN pip3 install --no-cache-dir --break-system-packages "numpy==2.1.2" && \
    pip3 install --no-cache-dir --break-system-packages "scikit-image==0.24.0" && \
    pip3 install --no-cache-dir --break-system-packages "matplotlib==3.10.8"

# Genesis (Quadrants compiler, amdgpu backend)
RUN pip3 install --no-cache-dir --break-system-packages "genesis-world==1.4.3"

# Quadrants dlopens "libamdhip64.so"; resolve it to the HIP runtime torch preloads from _rocm_sdk_core, otherwise a
# second copy gets loaded and kernel lookup fails (Genesis-Embodied-AI/genesis-world#3059)
# /opt/site-packages is the stable alias ryzer_env points at whichever directory
# actually holds the _rocm_sdk_* wheels. The old literal
# /opt/venv/lib/python3.12/site-packages path is wrong on the ROCm 10.1 base,
# where the SDK wheels live in the system dist-packages rather than the venv.
ENV ROCM_SDK_CORE_LIB=/opt/site-packages/_rocm_sdk_core/lib
# The HIP soname major follows the clr component, not the ROCm release, so read it off the installed runtime rather
# than hardcoding it (it was .so.7 under ROCm 7.14 and may or may not have moved under 10.1); a link pointing at a
# name that isn't there only shows up later as the same kernel lookup failure
RUN cd ${ROCM_SDK_CORE_LIB} && \
    hip_soname="$(ls libamdhip64.so.* | sort | head -n1)" && \
    ln -sf "$hip_soname" libamdhip64.so && \
    echo "libamdhip64.so -> $hip_soname"
ENV LD_LIBRARY_PATH=${ROCM_SDK_CORE_LIB}:${LD_LIBRARY_PATH}
# Quadrants links GPU kernels with ld.lld; don't rely on /opt/rocm being present
ENV PATH=${ROCM_SDK_CORE_LIB}/llvm/bin:${PATH}
# Force the monolithic constraint solver; the decomposed one stalls while compiling on AMD GPUs (Genesis-Embodied-AI/genesis-world#3545)
ENV QD_PERFDISPATCH_FORCE=func_solve_body:func_solve_body_monolith

# Copy test script
COPY test.py /ryzers/test_genesis.py
COPY demo.sh /ryzers/demo_genesis.sh
COPY demo.py /ryzers/demo_genesis.py
RUN chmod +x /ryzers/test_genesis.py /ryzers/demo_genesis.sh /ryzers/demo_genesis.py

RUN git clone --depth 1 --branch v1.4.3 https://github.com/Genesis-Embodied-AI/Genesis

CMD /ryzers/test_genesis.py 
