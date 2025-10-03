# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /ryzers

COPY . ./

RUN apt-get update && \
    apt-get install -y libopencv-dev cmake && \
    apt-get autoremove -y --purge && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/pierotofy/OpenSplat.git && \
    cd OpenSplat 

WORKDIR /ryzers/OpenSplat

SHELL ["/bin/bash", "-c"]

ENV HSA_OVERRIDE_GFX_VERSION=11.0.0

RUN mkdir build && cd build && \
    export PYTORCH_ROCM_ARCH=gfx1151 && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DGPU_RUNTIME=HIP \
        -DHIP_ROOT_DIR=/opt/rocm \
        -DHIP_PATH=/opt/rocm-7.0.0/lib/llvm \
        -DOPENSPLAT_BUILD_SIMPLE_TRAINER=ON \
        -DCMAKE_PREFIX_PATH=/opt/venv/lib/python3.12/site-packages/torch \
        -DTorch_DIR=/opt/venv/lib/python3.12/site-packages/torch/share/cmake/Torch \
        -DCMAKE_INSTALL_PREFIX=/ryzers/install && \
    make && make install

# Copy the test script
COPY test.sh /ryzers/opensplat_test.sh
RUN chmod +x /ryzers/opensplat_test.sh

# Set the entrypoint to the test script
CMD /ryzers/opensplat_test.sh

