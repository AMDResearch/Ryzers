# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /ryzers


ENV HSA_OVERRIDE_GFX_VERSION=11.0.0


RUN python -m pip install flwr

ENTRYPOINT ["flower-superlink"]
