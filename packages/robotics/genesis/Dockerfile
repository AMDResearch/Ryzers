# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}  

ENV DEBIAN_FRONTEND=noninteractive
ENV HSA_OVERRIDE_GFX_VERSION=11.0.0

WORKDIR /ryzers

# Install Vulkan
RUN wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | tee /etc/apt/trusted.gpg.d/lunarg.asc
RUN wget -qO /etc/apt/sources.list.d/lunarg-vulkan-1.4.309-jammy.list https://packages.lunarg.com/vulkan/1.4.309/lunarg-vulkan-1.4.309-jammy.list 
RUN apt update 
RUN apt install -y vulkan-sdk

# Necessary for visualisation on Wayland backed systems like Ubuntu
ENV PYOPENGL_PLATFORM='glx'

# Copy test script
COPY test.py /ryzers/test_genesis.py
RUN chmod +x /ryzers/test_genesis.py

RUN cd /ryzers

# Install Genesis verified against 0.2.1
RUN pip3 install genesis-world trimesh
RUN pip3 install libigl==2.5.1

CMD /ryzers/test_genesis.py 
