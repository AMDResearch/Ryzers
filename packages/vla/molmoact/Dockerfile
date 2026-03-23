# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /ryzers

# Download test image
RUN mkdir -p /ryzers/data/ && \
    wget https://huggingface.co/allenai/MolmoAct-7B-D-0812/resolve/main/example_1.png -O /ryzers/data/example_1.png && \
    wget https://huggingface.co/allenai/MolmoAct-7B-D-0812/resolve/main/example_2.png -O /ryzers/data/example_2.png

# Install python packages
RUN pip install einops accelerate "transformers==4.52"

# Copy test script
COPY test.py /ryzers/test_molmoact.py

CMD python /ryzers/test_molmoact.py
