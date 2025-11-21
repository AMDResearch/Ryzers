# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /ryzers

# Install deps
RUN apt-get update && apt-get install -y git vim wget

# OpenVLA is a dependency that will try to re-install pytorch
# we patch and install separately to preserve rocm-native install
RUN git clone https://github.com/openvla/openvla && \
    cd openvla && \
    sed -i -E '/^\s*dependencies\s*=\s*\[/,/^\s*\]/{ /^\s*"(torch|torchvision|torchaudio)[^"]*",?\s*(#.*)?$/d }' pyproject.toml && \
    pip install -e .

# Installed patched CogACT
RUN git clone https://github.com/microsoft/CogACT && \
    cd CogACT && \
    sed -i -E '/^\s*dependencies\s*=\s*\[/,/^\s*\]/{ /^\s*"(torch|torchvision|torchaudio)[^"]*",?\s*(#.*)?$/d }' pyproject.toml && \
    sed -i -E 's/"openvla @[^"]*"/"openvla"/' pyproject.toml && \
    pip install -e .

# Download test image
RUN mkdir -p /ryzers/data/ && \
    wget https://github.com/AMDResearch/Riallto/blob/main/notebooks/images/jpg/toucan.jpg?raw=true -O /ryzers/data/toucan.jpg

COPY test_cogact.py /ryzers/test_cogact.py
COPY test_cogact.sh /ryzers/test_cogact.sh
RUN chmod +x /ryzers/test_cogact.sh

# Create an entrypoint to automatically login to Hugging Face
# You need an account that has permissions to use the Llama2 models!!
RUN echo -e '#!/bin/bash\n\
if [ -n "$HF_TOKEN" ]; then\n\
  hf auth login --token "$HF_TOKEN" --add-to-git-credential\n\
fi\n\
export HF_USER=$(hf auth whoami | head -n 1 | cut -d" " -f3)\n\
echo "🚀 Logged in as $HF_USER"\n\
exec "$@"' > /ryzers/entrypoint.sh && chmod +x /ryzers/entrypoint.sh

ENTRYPOINT ["/ryzers/entrypoint.sh"]
CMD ["/bin/bash", "-c", "/ryzers/test_cogact.sh"]
