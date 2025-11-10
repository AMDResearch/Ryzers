# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Set environment variable for remote installation
ENV PYNQ_REMOTE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    jupyterlab \
    pillow \
    matplotlib \
    "numpy<2" \
    grpcio==1.64.0 \
    grpcio-tools==1.64.0 \
    wheel \
    pandas \
    scipy \
    seaborn \
    plotly \
    ipywidgets \
    tqdm \
    pynq

# Create a workspace directory
RUN mkdir -p /workspace
WORKDIR /workspace

# Copy test script
COPY test.sh /usr/local/bin/test.sh
RUN chmod +x /usr/local/bin/test.sh

# Expose ports for Jupyter and PYNQ.remote
EXPOSE 8888 7967

# Default command runs the test
CMD ["test.sh"]
