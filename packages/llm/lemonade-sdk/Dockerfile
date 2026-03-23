# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        wget \
        git

# Install lemonade-server via official Debian package
RUN wget -q -O /tmp/lemonade-server.deb \
        https://github.com/lemonade-sdk/lemonade/releases/latest/download/lemonade-server_10.0.0_amd64.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends /tmp/lemonade-server.deb \
    && rm -f /tmp/lemonade-server.deb \
    && rm -rf /var/lib/apt/lists/*

# Copy test script
WORKDIR /ryzers
COPY test.sh /ryzers/test_lemonade-sdk.sh
RUN chmod +x /ryzers/test_lemonade-sdk.sh

EXPOSE 8000

CMD /ryzers/test_lemonade-sdk.sh
