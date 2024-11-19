# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Set the environment variable required for HIP kernel compilation
ENV HSA_OVERRIDE_GFX_VERSION=11.0.0
    
WORKDIR /ryzers

RUN apt update
RUN apt upgrade -y
RUN apt install python3-pip

# Clone the repository and install segment-anything
RUN git clone https://github.com/ChaoningZhang/MobileSAM.git && \
    cd MobileSAM && \
    pip3 install -e . && \
    pip3 install timm gradio matplotlib opencv-python

RUN pip3 uninstall -y torch torchvision torchaudio
RUN pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.4/

COPY test.sh /ryzers/test_mobilesam.sh
RUN chmod +x /ryzers/test_mobilesam.sh

CMD /ryzers/test_mobilesam.sh

