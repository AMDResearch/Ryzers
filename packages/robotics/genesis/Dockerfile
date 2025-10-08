# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}  

ENV DEBIAN_FRONTEND=noninteractive
ENV HSA_OVERRIDE_GFX_VERSION=11.0.0

WORKDIR /ryzers

RUN apt-get install -y wget

# Install Vulkan
RUN wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | tee /etc/apt/trusted.gpg.d/lunarg.asc
RUN wget -qO /etc/apt/sources.list.d/lunarg-vulkan-noble.list http://packages.lunarg.com/vulkan/lunarg-vulkan-noble.list
RUN apt update 
RUN apt install -y vulkan-sdk

# Necessary for visualisation on Wayland backed systems like Ubuntu
ENV PYOPENGL_PLATFORM='glx'

# Copy test script
COPY test.py /ryzers/test_genesis.py
COPY demo.sh /ryzers/demo_genesis.sh
RUN chmod +x /ryzers/test_genesis.py /ryzers/demo_genesis.sh

RUN git clone https://github.com/Genesis-Embodied-AI/Genesis

# Install Genesis verified against 0.2.1
RUN pip3 install genesis-world trimesh
RUN pip3 install libigl==2.5.1
RUN pip3 install numpy==1.26.4

CMD /ryzers/test_genesis.py 
