# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# Install Python and Flower CLI
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
 && rm -rf /var/lib/apt/lists/*

# Install Flower CLI
RUN pip3 install --no-cache-dir --break-system-packages flwr[simulation]==1.27.0

# Copy test script
WORKDIR /ryzers
COPY test.py /ryzers/test_flower.py
RUN chmod +x /ryzers/test_flower.py

CMD ["/ryzers/test_flower.py"]
