# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Use non-login shell to prevent conda auto-activation.
# The base ROCm image ships system Python 3.12 with ROCm PyTorch;
# conda's py_3.10 env overrides it and must be removed.
SHELL ["/bin/bash", "-c"]

# Remove conda so every ryzer uses the system Python + ROCm PyTorch.
# Without this, login-shell profile scripts activate conda py_3.10
# which ships an older torch and incompatible numpy.
RUN if [ -d /opt/conda ]; then \
        rm -rf /opt/conda /opt/miniconda* && \
        find /etc/profile.d -name '*conda*' -delete 2>/dev/null || true && \
        sed -i '/conda/d' /root/.bashrc 2>/dev/null || true && \
        sed -i '/conda/d' /etc/bash.bashrc 2>/dev/null || true; \
    fi && hash -r

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
