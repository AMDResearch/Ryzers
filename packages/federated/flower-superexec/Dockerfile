# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /ryzers

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment with system site packages (inherit torch from ROCm base)
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv --system-site-packages $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install Flower and dependencies for the quickstart app
RUN pip install --no-cache-dir \
    "flwr>=1.26.0" \
    "flwr-datasets[vision]>=0.5.0" \
    "datasets"

# Copy the quickstart Flower project
COPY quickstart/ /ryzers/quickstart/

# Install the quickstart project
WORKDIR /ryzers/quickstart
RUN pip install --no-cache-dir -e .

WORKDIR /ryzers

# Copy test script
COPY test.sh /ryzers/test_flower_superexec.sh
RUN chmod +x /ryzers/test_flower_superexec.sh

# Default: show usage
CMD /ryzers/test_flower_superexec.sh
