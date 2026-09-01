# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /ryzers

COPY . ./

RUN apt-get update && \
    apt-get install -y libopencv-dev cmake curl unzip && \
    apt-get autoremove -y --purge && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/opensplat-test-data && \
    curl -L \
        'https://drive.usercontent.google.com/download?id=1mUUZFDo2swd6CE5vwPPkjN63Hyf4XyEv&export=download&confirm=t' \
        -o /tmp/banana.zip && \
    unzip -q /tmp/banana.zip -d /opt/opensplat-test-data && \
    rm /tmp/banana.zip

RUN git clone https://github.com/pierotofy/OpenSplat.git && \
    cd OpenSplat && \
    sed -i \
        's/c10::hip::HIPCachingAllocator/c10::cuda::CUDACachingAllocator/g' \
        model.cpp

WORKDIR /ryzers/OpenSplat

SHELL ["/bin/bash", "-c"]

ENV LD_LIBRARY_PATH=/opt/venv/lib/python3.13/site-packages/_rocm_sdk_devel/lib/host-math/lib:/opt/venv/lib/python3.13/site-packages/_rocm_sdk_core/lib/host-math/lib:/opt/venv/lib/python3.13/site-packages/_rocm_sdk_devel/lib/rocm_sysdeps/lib:/opt/venv/lib/python3.13/site-packages/_rocm_sdk_core/lib/rocm_sysdeps/lib:${LD_LIBRARY_PATH}

RUN mkdir build && cd build && \
    export PYTORCH_ROCM_ARCH=gfx1151 && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DGPU_RUNTIME=HIP \
        -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
        -DHIP_ROOT_DIR=/opt/rocm \
        -DHIP_PATH=/opt/rocm \
        -DOPENSPLAT_BUILD_SIMPLE_TRAINER=ON \
        -DCMAKE_PREFIX_PATH="/opt/rocm/lib/cmake;/opt/venv/lib/python3.13/site-packages/torch" \
        -DTorch_DIR=/opt/venv/lib/python3.13/site-packages/torch/share/cmake/Torch \
        -DCMAKE_INSTALL_PREFIX=/ryzers/install && \
    make && make install

# Copy the test script
COPY test.sh /ryzers/opensplat_test.sh
RUN chmod +x /ryzers/opensplat_test.sh

# Set the entrypoint to the test script
CMD /ryzers/opensplat_test.sh

