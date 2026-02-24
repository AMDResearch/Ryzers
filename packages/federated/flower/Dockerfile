# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /ryzers

RUN pip install flwr

RUN flwr new @flwrlabs/quickstart-pytorch

COPY test.sh .
RUN chmod +x test.sh
CMD ./test.sh
