# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}  

# Use an argument to get GPU architecture
ARG HSA_OVERRIDE_GFX_VERSION
ARG GFXSTRING

# Install dependencies
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libssl-dev \
    libcurl4-openssl-dev \
    rsync

#  Install Vulkan backend
RUN apt-get update \
&& wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | tee /etc/apt/trusted.gpg.d/lunarg.asc \
&& wget -qO /etc/apt/sources.list.d/lunarg-vulkan-noble.list https://packages.lunarg.com/vulkan/lunarg-vulkan-noble.list \
&& apt update \
&& apt install -y vulkan-sdk

# ROCm 7.14+ has no /opt/rocm; the HIP CMake package ships in rocm-sdk-devel.
RUN /opt/venv/bin/pip install \
      --index-url https://repo.amd.com/rocm/whl-multi-arch \
      --extra-index-url https://pypi.org/simple \
      "rocm[devel]==7.14.0" \
 && /opt/venv/bin/rocm-sdk init

ENV ROCM_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel

# The wheel lib dirs are not on the loader path, so binaries fail on libhipblas.
RUN printf '%s\n' \
      /opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib \
      /opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries/lib \
      /opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib \
      > /etc/ld.so.conf.d/rocm-wheels.conf && ldconfig

# Clone and build LlamaCPP
WORKDIR /ryzers
RUN git clone https://github.com/ggml-org/llama.cpp.git llamacpp && \
    cd llamacpp && \
    HIPCXX="$ROCM_PATH/lib/llvm/bin/clang++" HIP_PATH="$ROCM_PATH" \
    cmake -S . -B build -DGGML_VULKAN=1 -DGGML_HIP=ON -DGPU_TARGETS=$GFXSTRING \
    -DCMAKE_PREFIX_PATH="$(/opt/venv/bin/rocm-sdk path --cmake)" -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config Release -- -j 16

# Copy test script
COPY test.sh /ryzers/test_llamacpp.sh
RUN chmod +x /ryzers/test_llamacpp.sh

# Set entrypoint to test script
CMD ["/ryzers/test_llamacpp.sh"]
