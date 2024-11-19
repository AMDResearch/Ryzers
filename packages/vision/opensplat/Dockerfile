# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /ryzers

COPY . ./

RUN apt-get update && \
    apt-get install -y libopencv-dev && \
    apt-get autoremove -y --purge && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/pierotofy/OpenSplat.git && \
    cd OpenSplat 

WORKDIR /ryzers/OpenSplat

SHELL ["/bin/bash", "-c"]
RUN source activate py_3.12 && \
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DGPU_RUNTIME=HIP \
        -DHIP_ROOT_DIR=/opt/rocm \
        -DOPENSPLAT_BUILD_SIMPLE_TRAINER=ON \
        -DCMAKE_PREFIX_PATH=/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch \
        -DTorch_DIR=/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/share/cmake/Torch \
        -DCMAKE_INSTALL_PREFIX=/ryzers/install && \
    make && make install

# Copy the test script
COPY test.sh /ryzers/opensplat_test.sh
RUN chmod +x /ryzers/opensplat_test.sh

# Set the entrypoint to the test script
CMD /ryzers/opencv_opensplat.sh

