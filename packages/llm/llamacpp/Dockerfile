# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}  

# Use an argument to get GPU architecture
ARG HSA_OVERRIDE_GFX_VERSION

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

# Clone and build LlamaCPP
WORKDIR /ryzers
RUN git clone https://github.com/ggml-org/llama.cpp.git llamacpp && \
    cd llamacpp && \
    HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
    cmake -S . -B build -DLLAMA_CURL=ON -DGGML_HIP=ON -DAMDGPU_TARGETS=$HSA_OVERRIDE_GFX_VERSION -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config Release -- -j 16

# Copy test script
COPY test.sh /ryzers/test_llamacpp.sh
RUN chmod +x /ryzers/test_llamacpp.sh

# Set entrypoint to test script
CMD ["/ryzers/test_llamacpp.sh"]
