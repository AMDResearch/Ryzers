# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /ryzers

COPY quickstart-pytorch /ryzers/quickstart-pytorch 

ENTRYPOINT[flower-superlink]
