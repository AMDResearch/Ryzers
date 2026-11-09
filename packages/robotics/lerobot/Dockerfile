# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG LEROBOT_VERSION=v0.6.0

WORKDIR /ryzers

RUN apt-get update && apt-get install -y git cmake build-essential python3-dev pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev ffmpeg speech-dispatcher && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Lift lerobot's torch ceilings to admit the torch the base image ships. Bounds
# are derived from the installed versions rather than hardcoded -- this pin has
# needed a hand-edit on every base bump, and ROCm 10.1 ships a torch the previous
# ceiling excludes. The regex match replaces the old exact-string sed, which
# silently did nothing whenever upstream reworded its pin; the grep makes that
# failure loud instead.
RUN git clone --branch "${LEROBOT_VERSION}" --depth 1 https://github.com/huggingface/lerobot.git && \
    cd lerobot && \
    TORCH_MAX="$(python -c 'import torch; v=torch.__version__.split("+")[0].split("."); print(f"{int(v[0])}.{int(v[1])+1}.0")')" && \
    TV_MAX="$(python -c 'import torchvision; v=torchvision.__version__.split("+")[0].split("."); print(f"{int(v[0])}.{int(v[1])+1}.0")')" && \
    echo "lifting ceilings to torch<${TORCH_MAX} torchvision<${TV_MAX}" && \
    sed -i -E \
        -e "s/\"torch>=([0-9.]+),<[0-9.]+\"/\"torch>=\1,<${TORCH_MAX}\"/" \
        -e "s/\"torchvision>=([0-9.]+),<[0-9.]+\"/\"torchvision>=\1,<${TV_MAX}\"/" \
        -e 's/<0.12.0/<0.13.0/g' \
        pyproject.toml && \
    grep -qE "\"torch>=[0-9.]+,<${TORCH_MAX}\"" pyproject.toml && \
    grep -qE "\"torchvision>=[0-9.]+,<${TV_MAX}\"" pyproject.toml && \
    grep -nE '"torch(vision)?>=' pyproject.toml && \
    pip install --no-cache-dir -e ".[training,core_scripts,aloha,pusht,dynamixel,feetech,pi,diffusion]"

COPY test_lerobot.py /ryzers/test_lerobot.py
COPY test.sh /ryzers/test_lerobot.sh
COPY entrypoint.sh /ryzers/entrypoint.sh
RUN chmod +x /ryzers/test_lerobot.sh /ryzers/entrypoint.sh

ENTRYPOINT ["/ryzers/entrypoint.sh"]
CMD /ryzers/test_lerobot.sh
