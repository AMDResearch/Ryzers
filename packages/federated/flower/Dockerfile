# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /ryzers

RUN pip install -U "flwr[simulation]"

RUN flwr new @flwrlabs/quickstart-pytorch
RUN cd quickstart-pytorch && sed -i '/torch=/d;/torchvision=/d' pyproject.toml && pip install -e .

# Configure flwr to use local-deployment superlink
RUN mkdir -p ~/.flwr && \
    echo "[superlink.local-deployment]" > ~/.flwr/config.toml && \
    echo 'address = "127.0.0.1:9093"' >> ~/.flwr/config.toml && \
    echo "insecure = true" >> ~/.flwr/config.toml

COPY test.sh .
COPY run_quickstart.sh .
RUN chmod +x test.sh run_quickstart.sh
CMD ./run_quickstart.sh
