# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /ryzers

# Install deps
RUN apt-get update && apt-get install -y git vim wget

# Download test image
RUN mkdir -p /ryzers/data/ && \
    wget https://github.com/AMDResearch/Riallto/blob/main/notebooks/images/jpg/toucan.jpg?raw=true -O /ryzers/data/toucan.jpg

RUN pip install -r https://raw.githubusercontent.com/openvla/openvla/refs/heads/main/requirements-min.txt

RUN pip install accelerate

COPY test.py /ryzers/test.py

COPY test.sh /ryzers/test_openvla.sh
RUN chmod +x /ryzers/test_openvla.sh

CMD ["/bin/bash", "-c", "/ryzers/test_openvla.sh"]
