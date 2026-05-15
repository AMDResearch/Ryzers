# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}
SHELL ["/bin/bash", "-c"]

WORKDIR /ryzers

RUN apt-get update && apt-get install -y git cmake build-essential python3-dev pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev pkg-config ffmpeg speech-dispatcher && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN git config --global credential.helper store

# Install release v0.4
RUN git clone https://github.com/huggingface/lerobot.git \
 && cd lerobot \
 && git checkout f25ac02 \
 && sed -i '/"torch[>=<~ ]*/d' pyproject.toml \
 && sed -i '/"torchvision[>=<~ ]*/d' pyproject.toml \ 
 && pip install -e ".[aloha,pusht,dynamixel,feetech,pi]"

RUN pip install --no-cache-dir "numpy<2" pytest transformers huggingface_hub num2words

COPY test.sh /ryzers/test_lerobot.sh
RUN chmod +x /ryzers/test_lerobot.sh

# Create an entrypoint to autmatically login to Hugging Face on docker start
RUN echo -e '#!/bin/bash\n\
if [ -n "$HF_TOKEN" ]; then\n\
  hf auth login --token "$HF_TOKEN" --add-to-git-credential\n\
fi\n\
export HF_USER=$(hf auth whoami | head -n 1 | cut -d" " -f3)\n\
echo "🚀 Logged in as $HF_USER"\n\
exec "$@"' > /ryzers/entrypoint.sh && chmod +x /ryzers/entrypoint.sh

ENTRYPOINT ["/ryzers/entrypoint.sh"]
CMD /ryzers/test_lerobot.sh
