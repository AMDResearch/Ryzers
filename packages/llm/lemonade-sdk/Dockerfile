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

# Install lemonade-sdk from PyPI. The Debian release was discontinued in v10.x
# upstream (only .pkg / .rpm / .tar.gz / .msi are published now), so the
# previous releases/latest/download/lemonade-server_10.0.0_amd64.deb URL
# 404s. PyPI is the canonical Linux distribution channel.
RUN python3 -m pip install --no-cache-dir --break-system-packages "lemonade-sdk"

# Copy test script
WORKDIR /ryzers
COPY test.sh /ryzers/test_lemonade-sdk.sh
RUN chmod +x /ryzers/test_lemonade-sdk.sh

EXPOSE 8000

CMD /ryzers/test_lemonade-sdk.sh
