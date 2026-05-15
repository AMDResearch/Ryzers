# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /ryzers

# Install Vulkan tools
RUN wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo tee /etc/apt/trusted.gpg.d/lunarg.asc && \
    wget -qO /etc/apt/sources.list.d/lunarg-vulkan-noble.list http://packages.lunarg.com/vulkan/lunarg-vulkan-noble.list && \
    apt-get update

RUN apt-get update && \
    apt-get -y install cmake vulkan-tools vulkan-sdk

# Build ncnn
RUN git clone https://github.com/Tencent/ncnn

RUN cd ncnn && \
    mkdir -p build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=ON -DNCNN_SYSTEM_GLSLANG=ON -DNCNN_BUILD_EXAMPLES=ON .. && \
    make -j$(nproc)

# Copy the test script
COPY test.sh /ryzers/test.sh
RUN chmod +x /ryzers/test.sh

# Copy benchmark script
COPY benchmark.sh /ryzers/benchmark.sh
RUN chmod +x /ryzers/benchmark.sh

# Set the entrypoint to the test script
# Validate ROCm torch
RUN /ryzers/test_init.sh

CMD /ryzers/test.sh

