# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /ryzers

# Download test image
RUN mkdir -p /ryzers/data/ && \
    wget https://github.com/AMDResearch/Riallto/blob/main/notebooks/images/jpg/toucan.jpg?raw=true -O /ryzers/data/toucan.jpg && \
    wget https://upload.wikimedia.org/wikipedia/commons/b/b0/Barbara_Sahakian_BBC_Radio4_The_Life_Scientific_29_May_2012_b01j5j24.flac -O /ryzers/data/audio.flac

# Install python packages
RUN pip install 'transformers==4.48.2' soundfile scipy peft backoff

# Copy test script
COPY test.py /ryzers/test_phi4.py

CMD python /ryzers/test_phi4.py
