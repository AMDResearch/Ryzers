# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG LEROBOT_VERSION=v0.6.0

WORKDIR /ryzers

RUN apt-get update && apt-get install -y git cmake build-essential python3-dev pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev ffmpeg speech-dispatcher bazel-bootstrap && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# labmaze (pulled in transitively via the aloha/pusht extras -> dm_control) has no
# cp313 wheel and builds from source with Bazel. Its bundled abseil references
# @platforms//cpu:wasm32, which the platforms repo it pins is too old to declare.
# Override platforms with a modern checkout and install labmaze up front so the
# lerobot install below reuses it instead of failing the Bazel build.
RUN git clone https://github.com/google-deepmind/labmaze /tmp/labmaze && \
    git clone --depth 1 https://github.com/bazelbuild/platforms /tmp/platforms && \
    echo 'build --override_repository=platforms=/tmp/platforms' >> /tmp/labmaze/.bazelrc && \
    pip install /tmp/labmaze

RUN git clone --branch "${LEROBOT_VERSION}" --depth 1 https://github.com/huggingface/lerobot.git && \
    cd lerobot && \
    sed -i \
        -e 's/"torch>=2.7,<2.12.0"/"torch>=2.7,<2.13.0"/' \
        -e 's/"torchvision>=0.22.0,<0.27.0"/"torchvision>=0.22.0,<0.28.0"/' \
        -e 's/<0.12.0/<0.13.0/g' \
        pyproject.toml && \
    pip install --no-cache-dir -e ".[training,core_scripts,aloha,pusht,dynamixel,feetech,pi,diffusion]"

COPY test_lerobot.py /ryzers/test_lerobot.py
COPY test.sh /ryzers/test_lerobot.sh
COPY entrypoint.sh /ryzers/entrypoint.sh
RUN chmod +x /ryzers/test_lerobot.sh /ryzers/entrypoint.sh

ENTRYPOINT ["/ryzers/entrypoint.sh"]
CMD /ryzers/test_lerobot.sh
