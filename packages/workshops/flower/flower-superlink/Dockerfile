# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /ryzers

# Install Python dependencies without torch (already in base image)
RUN pip install --no-cache-dir \
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
RUN pip install --no-deps --no-cache-dir \
    flwr==1.27.0

# Copy test script
COPY test.py /ryzers/test_superlink.py
RUN chmod +x /ryzers/test_superlink.py

# SuperLink ports: 9091 (SuperExec), 9092 (SuperNode), 9093 (REST API)
EXPOSE 9091 9092 9093

CMD ["/ryzers/test_superlink.py"]
