# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /ryzers

# Install flwr with all dependencies (SuperLink doesn't need torch, no conflict)
RUN pip install --no-cache-dir \
    flwr==1.27.0

# Copy test script
COPY test.py /ryzers/test_superlink.py
RUN chmod +x /ryzers/test_superlink.py

# SuperLink ports: 9091 (SuperExec), 9092 (SuperNode), 9093 (REST API)
EXPOSE 9091 9092 9093

CMD ["/ryzers/test_superlink.py"]
