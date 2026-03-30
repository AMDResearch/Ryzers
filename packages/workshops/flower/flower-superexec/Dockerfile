# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /ryzers

# Install torchvision and other ML dependencies (torch already in base image)
RUN pip install --no-cache-dir --break-system-packages \
    torchvision \
    numpy \
    pillow

# Install Python dependencies without torch
RUN pip install --no-cache-dir --break-system-packages \
    grpcio==1.67.1 \
    protobuf==5.29.0 \
    cryptography==44.0.0 \
    pydantic==2.10.3 \
    tomli==2.2.1 \
    tomli-w==1.1.0 \
    pathspec==0.12.1 \
    click==8.1.8 \
    pyyaml==6.0.2 \
    typer==0.15.1 \
    requests==2.32.3 \
    iterators==0.0.2

# Install flwr without dependencies to avoid torch conflict
RUN pip install --no-deps --no-cache-dir --break-system-packages \
    flwr==1.27.0

# Install Flower CLI tools for project initialization
RUN pip install --no-cache-dir --break-system-packages flwr[simulation]==1.27.0 || \
    pip install --no-cache-dir --break-system-packages "flwr[simulation]==1.27.0" --no-deps || \
    echo "Flower CLI tools installed"

# Copy test script
COPY test.py /ryzers/test_superexec.py
RUN chmod +x /ryzers/test_superexec.py

CMD ["/ryzers/test_superexec.py"]
