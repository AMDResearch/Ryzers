# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /ryzers

# Install deps
RUN apt-get update && apt-get install -y git \
	vim \
	wget \
	libegl1-mesa-dev \
	libgles2-mesa-dev

RUN git clone https://github.com/tonyzhaozh/act

RUN pip install matplotlib && \
	pip install einops && \
	pip install packaging && \
	pip install h5py && \
 	pip install ipython && \
	pip install dm_control && \
	pip install pyquaternion


COPY test.sh /ryzers/
RUN chmod +x /ryzers/

# Validate ROCm torch
RUN /ryzers/test_init.sh

CMD ["/bin/bash", "-c", "/ryzers/test.sh"]
