# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-lc"]

RUN getent group render || groupadd render
RUN getent group video || groupadd video
RUN getent group kvm || groupadd kvm

RUN usermod -aG render root
RUN usermod -aG video root
RUN usermod -aG kvm root

WORKDIR /ryzers
COPY test.sh /ryzers/test_init.sh
RUN chmod +x /ryzers/test_init.sh

CMD /ryzers/test_init.sh
