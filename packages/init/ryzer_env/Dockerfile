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

# Normalise the Python layout before anything else depends on it.
# The ROCm 7.x images shipped a virtualenv at /opt/venv with `python` on PATH.
# The ROCm 10.1 images dropped it: there is no /opt/venv, no `python` (only
# `python3`), and packages live in /usr/local/lib/pythonX.Y/dist-packages.
# Almost every Ryzer package assumes the old contract, and worse, the version
# probe below is guarded by `|| exit 0` -- with no `python` it would succeed
# while silently skipping the entire ROCm devel install, leaving downstream HIP
# builds with no toolchain and no error. Recreate the contract here, in the one
# layer every chain already starts from, rather than editing 40+ Dockerfiles.
# --system-site-packages keeps the base image's torch visible inside the venv.
# The guard makes this a no-op on images that already provide /opt/venv.
RUN if [ ! -x /opt/venv/bin/python ]; then \
        (python3 -m venv --system-site-packages /opt/venv 2>/dev/null || \
         { apt-get update && apt-get install -y --no-install-recommends python3-venv && \
           rm -rf /var/lib/apt/lists/* && python3 -m venv --system-site-packages /opt/venv; }) ; \
    fi && \
    /opt/venv/bin/python --version
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:${PATH}

# Add HIP build tools when using wheel-based ROCm.
# Two indexes, because they serve disjoint ROCm ranges and packages may pin either:
# whl-next has 10.0.0 and 10.1.0 only; whl-multi-arch stops at 7.14.1. Querying both
# keeps packages that pin an older base (capx, gap) building after the default moved
# to ROCm 10.1. Re-check both on each ROCm bump -- a version served by neither makes
# the pip install fail, and the chain then dies at `rocm-sdk init` rather than here.
RUN ROCM_VERSION="$(python -c "import importlib.metadata as m; print(m.version('rocm'))" 2>/dev/null)" || exit 0; \
    pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ \
                --extra-index-url https://repo.amd.com/rocm/whl-multi-arch/ \
                "rocm[devel]==${ROCM_VERSION}"; \
    rocm-sdk init; \
    test -e /opt/rocm || ln -s "$(rocm-sdk path --root)" /opt/rocm; \
    ln -sfn "$(dirname "$(rocm-sdk path --root)")" /opt/site-packages; \
    echo "ROCm SDK wheels: $(readlink -f /opt/site-packages)"

# /opt/site-packages (created above) is a stable alias for whichever directory
# actually holds the _rocm_sdk_* wheels. LD_LIBRARY_PATH used to spell out
# /opt/venv/lib/python3.12/site-packages, which has needed a hand-edit on every
# upgrade and is simply wrong on ROCm 10.1: the SDK wheels live in the system
# dist-packages there, not the venv, because `rocm[devel]` is already satisfied
# from the base image and pip installs nothing into the venv. Deriving the path
# from `rocm-sdk path --root` is correct on both layouts.
ENV ROCM_HOME=/opt/rocm
ENV LD_LIBRARY_PATH=/opt/site-packages/_rocm_sdk_devel/lib:/opt/site-packages/_rocm_sdk_core/lib:/opt/site-packages/_rocm_sdk_libraries/lib:${LD_LIBRARY_PATH}

# Sign-of-life test
WORKDIR /ryzers
COPY test.sh /ryzers/test_init.sh
RUN chmod +x /ryzers/test_init.sh

CMD /ryzers/test_init.sh  
