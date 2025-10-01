# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# add dependencies
RUN apt-get update && \
    apt-get install -y \
	konsole 


ENV AMDGPUTOP_TIMEOUT=10s

RUN wget https://github.com/Umio-Yasuno/amdgpu_top/releases/download/v0.11.0/amdgpu-top_0.11.0-1_amd64.deb \
    -O amdgpu-top.deb
RUN dpkg --install amdgpu-top.deb



# add a sign-of-life test
WORKDIR /ryzers
COPY test.sh /ryzers/test_amdgputop.sh
RUN chmod +x /ryzers/test_amdgputop.sh

# Command to run a test
CMD /ryzers/test_amdgputop.sh
