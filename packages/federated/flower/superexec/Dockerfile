# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV HSA_OVERRIDE_GFX_VERSION=11.0.0

WORKDIR /ryzers

RUN python -m pip install flwr

# Create and configure the quickstart project
RUN flwr new @flwrlabs/quickstart-pytorch
RUN cd quickstart-pytorch && \
    sed -i '/torch=/d;/torchvision=/d' pyproject.toml && \
    sed -i 's/.*flwr\[simulation\].*//' pyproject.toml && \
    python -m pip install -U --no-cache-dir .

ENTRYPOINT ["flower-superexec"]
