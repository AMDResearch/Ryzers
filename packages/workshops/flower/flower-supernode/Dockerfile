# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /ryzers

# Install flwr with all dependencies (SuperNode doesn't need torch, no conflict)
RUN pip install --no-cache-dir \
    flwr==1.27.0

# Copy test script
COPY test.py /ryzers/test_supernode.py
RUN chmod +x /ryzers/test_supernode.py

# SuperNode ClientApp I/O port
EXPOSE 9094

CMD ["/ryzers/test_supernode.py"]
