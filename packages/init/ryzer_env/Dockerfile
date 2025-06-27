# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# handle add the render group
RUN groupadd --system render
RUN groupadd --system kvm

RUN usermod -aG render root
RUN usermod -aG video root
RUN usermod -aG kvm root

# add a sign-of-life test
WORKDIR /ryzers
COPY test.sh /ryzers/test_init.sh
RUN chmod +x /ryzers/test_init.sh

# Command to run a test
CMD /ryzers/test_init.sh  