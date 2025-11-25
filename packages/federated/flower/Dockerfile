# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /ryzers

# Download test image
RUN mkdir -p /ryzers/data/ && \
	pip3 install flwr

# Copy test script
COPY test.sh /ryzers/test.sh
RUN chmod +x /ryzers/test.sh
CMD /ryzers/test.sh
