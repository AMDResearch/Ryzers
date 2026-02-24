# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /ryzers

RUN pip install -U "flwr[simulation]"

RUN flwr new @flwrlabs/quickstart-pytorch
RUN cd quickstart-pytorch && sed -i '/torch=/d;/torchvision=/d' pyproject.toml && pip install -e .


COPY test.sh .
RUN chmod +x test.sh
CMD ./test.sh
