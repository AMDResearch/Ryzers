# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Set environment variables to prevent Python from writing .pyc files and buffering output
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /ryzers/notebooks

# Install system dependencies for JupyterLab
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install JupyterLab
RUN python3 -m pip install --no-cache-dir jupyterlab

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy test script
COPY test.sh /ryzers/test_jupyterlab.sh
RUN chmod +x /ryzers/test_jupyterlab.sh

# Expose the default JupyterLab port
EXPOSE 8888

# Run test by default (use 'docker run ... jupyter-lab --ip=0.0.0.0 --no-browser --allow-root' for interactive use)
CMD ["/ryzers/test_jupyterlab.sh"]

