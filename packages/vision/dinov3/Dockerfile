# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /ryzers

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Configure git credential storage for huggingface-cli
RUN git config --global credential.helper store

# Install required Python packages
RUN pip install --no-cache-dir \
    "transformers>=4.56.0" \
    accelerate \
    "numpy<2" \
    pillow \
    requests \
    huggingface_hub \
    matplotlib

# Download test image
RUN mkdir -p /ryzers/data/ && \
    wget https://github.com/AMDResearch/Riallto/blob/main/notebooks/images/jpg/toucan.jpg?raw=true -O /ryzers/data/toucan.jpg

# Copy test scripts
COPY test_dinov3.py /ryzers/test_dinov3.py
COPY test.sh /ryzers/test_dinov3.sh
RUN chmod +x /ryzers/test_dinov3.sh

# Create an entrypoint to automatically login to Hugging Face
# You need an account that has permissions to use the DinoV3 models!!
RUN echo -e '#!/bin/bash\n\
if [ -n "$HF_TOKEN" ]; then\n\
  hf auth login --token "$HF_TOKEN" --add-to-git-credential\n\
fi\n\
export HF_USER=$(hf auth whoami | head -n 1 | cut -d" " -f3)\n\
echo "🚀 Logged in as $HF_USER"\n\
exec "$@"' > /ryzers/entrypoint.sh && chmod +x /ryzers/entrypoint.sh

ENTRYPOINT ["/ryzers/entrypoint.sh"]
CMD /ryzers/test_dinov3.sh
