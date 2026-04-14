# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /ryzers

# Install Flower and dependencies for the quickstart app
RUN pip install --no-cache-dir --upgrade \
    flwr \
    "flwr-datasets[vision]" \
    datasets

# Copy the quickstart Flower project
COPY quickstart/ /ryzers/quickstart/

# Add quickstart package to PYTHONPATH
ENV PYTHONPATH="/ryzers/quickstart:$PYTHONPATH"

# Pre-download CIFAR-10 dataset during build to avoid timeout during runtime
RUN python3 -c "from datasets import load_dataset; load_dataset('uoft-cs/cifar10')"

# Copy test script
COPY test.sh /ryzers/test_flower_superexec.sh
RUN chmod +x /ryzers/test_flower_superexec.sh

# Default: show usage
CMD /ryzers/test_flower_superexec.sh
