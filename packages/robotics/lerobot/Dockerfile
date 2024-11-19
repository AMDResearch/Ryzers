# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /ryzers

RUN apt-get update && apt-get install -y git cmake build-essential python3-dev pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev pkg-config ffmpeg speech-dispatcher && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN git config --global credential.helper store

RUN git clone https://github.com/huggingface/lerobot.git \
 && cd lerobot \
 && git checkout 2b71789 \
 && sed -i '/"torch[>=<~ ]*/d' pyproject.toml \
 && sed -i '/"torchvision[>=<~ ]*/d' pyproject.toml \ 
 && pip install -e ".[aloha,pusht,dynamixel]"

RUN pip install --no-cache-dir "numpy<2" pytest transformers huggingface_hub

# Patch lerobot
RUN printf '%s\n' \
    'from transformers import modeling_utils' \
    'if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:' \
    '    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none","colwise","rowwise"]' \
    | sed -i '51r /dev/stdin' /ryzers/lerobot/lerobot/common/policies/pi0/modeling_pi0.py
RUN sed -i 's/tasks = \[task if task\.endswith("\\n") else f"{task}\\n" for task in tasks\]/tasks = [tasks if tasks.endswith("\\n") else f"{tasks}\\n"]/' /ryzers/lerobot/lerobot/common/policies/pi0/modeling_pi0.py

COPY test.sh /ryzers/test_lerobot.sh
RUN chmod +x /ryzers/test_lerobot.sh

# Create an entrypoint to autmatically login to Hugging Face on docker start
RUN echo '#!/bin/bash\n\
if [ -n "$HF_TOKEN" ]; then\n\
  huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential\n\
fi\n\
export HF_USER=$(huggingface-cli whoami | head -n1)\n\
echo "🚀 Logged in as $HF_USER"\n\
exec "$@"' > /ryzers/entrypoint.sh && chmod +x /ryzers/entrypoint.sh

ENTRYPOINT ["/ryzers/entrypoint.sh"]
CMD /ryzers/test_lerobot.sh
