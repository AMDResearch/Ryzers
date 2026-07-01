# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE=ros
FROM ${BASE_IMAGE}

ARG ROS_DISTRO=jazzy
ENV ROS_DISTRO=${ROS_DISTRO}
ENV DEBIAN_FRONTEND=noninteractive

# ROS2 dependencies. Graphics packages (libgl1-mesa-dri + vulkan) added for the
# O3DE renderer, which the base image doesn't fully provide
RUN apt-get update && apt-get install -y --no-install-recommends \
    git unzip ffmpeg libportaudio2 python3-venv \
    libgl1-mesa-dri libvulkan1 mesa-vulkan-drivers vulkan-tools \
    ros-${ROS_DISTRO}-rai-interfaces \
    ros-${ROS_DISTRO}-moveit \
    ros-${ROS_DISTRO}-gazebo-msgs \
    ros-${ROS_DISTRO}-moveit-resources-panda-moveit-config \
    ros-${ROS_DISTRO}-depth-image-proc \
    ros-${ROS_DISTRO}-ackermann-msgs \
    ros-${ROS_DISTRO}-sdformat-urdf \
 && rm -rf /var/lib/apt/lists/*

# Rust toolchain (needed by rai_interfaces__rs ROS2 package)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable --no-modify-path
ENV PATH=/root/.cargo/bin:$PATH

RUN echo "/opt/ros/${ROS_DISTRO}/opt/sdformat_vendor/lib" > /etc/ld.so.conf.d/ros-sdformat.conf && \
    ldconfig

# Activate base image venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN echo "source $VIRTUAL_ENV/bin/activate" >> /root/.bashrc

# Pin empy<4 (required by rosidl_generator_rs)
RUN pip install --upgrade pip "setuptools<82" poetry "empy<4" lark && \
    pip install --no-build-isolation openai-whisper==20231117

# RAI framework (pinned commit)
WORKDIR /ryzers
RUN git clone https://github.com/RobotecAI/rai.git

WORKDIR /ryzers/rai
RUN git checkout a097617
RUN vcs import < ros_deps.repos
RUN vcs import < demos.repos

RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction

RUN rosdep init && \
    rosdep update && \
    rosdep install --from-paths src/examples/rai-manipulation-demo/ros2_ws/src --ignore-src -r -y

RUN . ./scripts/download_demo.sh manipulation

RUN . /opt/ros/${ROS_DISTRO}/setup.sh && \
    colcon build --symlink-install

# rai_bench / rai_sim live in the repo but aren't declared as poetry/colcon
# deps, so install from source; rai_perception comes from PyPI. It must go
# before the torch re-pin below (it would otherwise pull CUDA torch back in)
RUN pip install src/rai_bench && \
    pip install src/rai_sim && \
    pip install rai-perception

# Pre-download detection/segmentation weights (~1.5 GB) via rai-perception's own
# downloader, so model/URL bumps flow through automatically. Lands in the default
# ~/.cache/rai/vision/weights/ the services read at runtime.
# NOTE: relies on the `download-perception-models` console script staying stable.
RUN . /opt/ros/${ROS_DISTRO}/setup.sh && \
    . install/setup.sh && \
    download-perception-models

# Reinstall ROCm torch (poetry deps pull CUDA torch)
RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm7.2/ \
    --force-reinstall --no-deps

# Swap the CUDA triton (pulled with torch above) for the ROCm build; the CUDA
# one lacks triton.backends, which ROCm torch imports via inductor
RUN pip uninstall -y triton && \
    pip install pytorch-triton-rocm \
        --index-url https://download.pytorch.org/whl/rocm7.2/ --no-deps

WORKDIR /ryzers

COPY test.sh /ryzers/test_rai.sh
RUN chmod +x /ryzers/test_rai.sh
COPY manipulation_demo.sh /ryzers/manipulation_demo.sh
RUN chmod +x /ryzers/manipulation_demo.sh
COPY lemonade_env.sh /ryzers/

EXPOSE 8501
CMD /ryzers/test_rai.sh