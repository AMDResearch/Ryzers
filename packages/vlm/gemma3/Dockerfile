# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /ryzers

# Download test image
RUN mkdir -p /ryzers/data/ && \
    wget https://github.com/AMDResearch/Riallto/blob/main/notebooks/images/jpg/toucan.jpg?raw=true -O /ryzers/data/toucan.jpg

# Install python packages (for HF API)
RUN pip install -U transformers accelerate

# Copy test script
COPY test.sh /ryzers/test_gemma3.sh
COPY test_hf.py /ryzers/test_gemma3_hf.py
COPY demo.sh /ryzers/demo_gemma3.sh
RUN chmod +x /ryzers/test_gemma3.sh /ryzers/demo_gemma3.sh

CMD /ryzers/test_gemma3.sh
