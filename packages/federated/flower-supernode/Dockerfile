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

# Create virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install Flower
RUN pip install --no-cache-dir "flwr>=1.26.0"

# Copy test script
COPY test.sh /ryzers/test_flower_supernode.sh
RUN chmod +x /ryzers/test_flower_supernode.sh

# Expose ClientAppIO API port (default 9094, can be overridden)
EXPOSE 9094

# Default: show usage (user must specify superlink address and partition config)
CMD /ryzers/test_flower_supernode.sh
