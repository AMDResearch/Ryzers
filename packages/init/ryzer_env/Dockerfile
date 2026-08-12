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

# Add HIP build tools when using wheel-based ROCm.
RUN ROCM_VERSION="$(python -c "import importlib.metadata as m; print(m.version('rocm'))" 2>/dev/null)" || exit 0; \
    pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ "rocm[devel]==${ROCM_VERSION}"; \
    rocm-sdk init; \
    test -e /opt/rocm || ln -s "$(rocm-sdk path --root)" /opt/rocm

ENV ROCM_HOME=/opt/rocm
ENV LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib:/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib:/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries/lib:${LD_LIBRARY_PATH}

# Sign-of-life test
WORKDIR /ryzers
COPY test.sh /ryzers/test_init.sh
RUN chmod +x /ryzers/test_init.sh

CMD /ryzers/test_init.sh  
