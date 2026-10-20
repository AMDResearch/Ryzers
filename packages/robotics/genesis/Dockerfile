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
 && rm -rf /var/lib/apt/lists/*

# Locale setup (optional but kept for completeness)
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && locale-gen

# Genesis 1.4.x (pulls in the quadrants compute backend, python>=3.10,<3.14).
# PyTorch ROCm comes from the base image and is left untouched.
# The genesis-world 1.4.4 wheel fails in scene.build() on every AMD GPU (Genesis issue #3544,
# fixed by PR #3545), so install a main-branch commit that includes the fix. Switch back to
# "genesis-world==<version>" once a release newer than 1.4.4 is out.
ARG GENESIS_REF=a5c4e9a73aca221fa26e611281f2696c5f8b0ca8
RUN pip3 install --no-cache-dir --break-system-packages \
    "genesis-world @ git+https://github.com/Genesis-Embodied-AI/genesis-world@${GENESIS_REF}" \
    "matplotlib>=3.7"

# Examples matching the installed commit (tactile sensors, tutorials, ...)
RUN git clone https://github.com/Genesis-Embodied-AI/genesis-world Genesis && \
    git -C Genesis checkout "${GENESIS_REF}"

# Genesis/quadrants GPU settings on ROCm:
# - The wheel-based ROCm image ships two copies of libamdhip64 (_rocm_sdk_core for torch and
#   _rocm_sdk_devel, which comes first on LD_LIBRARY_PATH). Quadrants would load the devel copy next to
#   torch's, and its runtime module then fails with "Cannot look up function
#   runtime_get_rand_states_buffer_size". Preloading torch's copy makes both use the same HIP runtime.
# - Zero-copy torch<->quadrants tensor sharing segfaults in scene.build() with ROCm 7.14 (gfx1151), so disable it
#   (slower host/device transfers, but correct).
# - The default decomposed (graph) constraint solver did not finish compiling on gfx1151, so force the monolithic one.
ENV LD_PRELOAD=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib/libamdhip64.so.7 \
    GS_ENABLE_ZEROCOPY=0 \
    QD_PERFDISPATCH_FORCE=func_solve_body:func_solve_body_monolith

# Copy test script
COPY test.py /ryzers/test_genesis.py
COPY demo.sh /ryzers/demo_genesis.sh
COPY demo.py /ryzers/demo_genesis.py
RUN chmod +x /ryzers/test_genesis.py /ryzers/demo_genesis.sh /ryzers/demo_genesis.py

CMD /ryzers/test_genesis.py
