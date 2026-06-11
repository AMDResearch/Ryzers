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

# Install lemonade-server from PPA.
RUN add-apt-repository ppa:lemonade-team/stable
RUN apt-get install -y lemonade-server

# Copy test script
WORKDIR /ryzers
COPY test.sh /ryzers/test_lemonade-sdk.sh
RUN chmod +x /ryzers/test_lemonade-sdk.sh

EXPOSE 13305

CMD /ryzers/test_lemonade-sdk.sh
