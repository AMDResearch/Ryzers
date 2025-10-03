# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-lc"]


RUN getent group render || groupadd render
RUN getent group video || groupadd video
RUN getent group kvm || groupadd kvm


RUN usermod -aG render root
RUN usermod -aG video root
RUN usermod -aG kvm root

#################################
# Install ROCm Using ROCm Docker
#################################




#####################################
# Install ROCm Using TheRock Steps
#####################################

# add dependencies
# RUN apt-get update && \
#     apt-get install -y \
# 	python3 \
#     	python3.12 \
# 	python3.12-dev \
# 	python3.12-venv \
# 	python3-pip

# RUN python3 -m venv /ryzers/venv_ryzers
# ENV PATH="/ryzers/venv_ryzers/bin:$PATH"

# RUN  python3 -m pip install \
#   --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
#   rocm[libraries,devel]

# # install torch
# RUN  python3 -m pip install \
#   --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
#   torch torchaudio torchvision



# add a sign-of-life test
WORKDIR /ryzers
COPY test.sh /ryzers/test_init.sh
RUN chmod +x /ryzers/test_init.sh

# Command to run a test
CMD /ryzers/test_init.sh  
