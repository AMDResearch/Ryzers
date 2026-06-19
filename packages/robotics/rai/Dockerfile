# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE=ros
FROM ${BASE_IMAGE}

ARG ROS_DISTRO=jazzy
ENV ROS_DISTRO=${ROS_DISTRO}
ENV DEBIAN_FRONTEND=noninteractive

# ROS2 and graphics dependencies.
# libgl1-mesa-dri is required by the O3DE renderer (the base image ships only
# the Vulkan loader/drivers, not the Mesa DRI driver).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        libgl1-mesa-dri \
        libportaudio2 \
        libvulkan1 \
        mesa-vulkan-drivers \
        python3-venv \
        ros-${ROS_DISTRO}-ackermann-msgs \
        ros-${ROS_DISTRO}-depth-image-proc \
        ros-${ROS_DISTRO}-gazebo-msgs \
        ros-${ROS_DISTRO}-moveit \
        ros-${ROS_DISTRO}-moveit-resources-panda-moveit-config \
        ros-${ROS_DISTRO}-rai-interfaces \
        ros-${ROS_DISTRO}-sdformat-urdf \
        unzip \
        vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

# Rust toolchain (needed by the rai_interfaces__rs ROS2 package).
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --no-modify-path
ENV PATH=/root/.cargo/bin:$PATH

RUN echo "/opt/ros/${ROS_DISTRO}/opt/sdformat_vendor/lib" > /etc/ld.so.conf.d/ros-sdformat.conf \
    && ldconfig

# Activate the base image venv.
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"
RUN echo "source ${VIRTUAL_ENV}/bin/activate" >> /root/.bashrc

# Pin empy<4 (required by rosidl_generator_rs).
RUN pip install --upgrade pip "setuptools<82" poetry "empy<4" lark \
    && pip install --no-build-isolation openai-whisper==20231117

# RAI framework (pinned commit).
WORKDIR /ryzers
RUN git clone https://github.com/RobotecAI/rai.git

WORKDIR /ryzers/rai
RUN git checkout a097617
RUN vcs import < ros_deps.repos
RUN vcs import < demos.repos
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction
RUN rosdep init \
    && rosdep update \
    && rosdep install \
        --from-paths src/examples/rai-manipulation-demo/ros2_ws/src \
        --ignore-src -r -y
RUN . ./scripts/download_demo.sh manipulation
RUN . /opt/ros/${ROS_DISTRO}/setup.sh \
    && colcon build --symlink-install

# rai_bench is a Python package in the repo but not declared as a poetry/colcon dep.
RUN pip install -e src/rai_bench

# rai_sim provides the simulation bridge (O3DE connector) used by the
# manipulation demo/benchmark. Like rai_bench it lives in the repo but isn't
# declared as a poetry/colcon dependency, so install it explicitly.
RUN pip install -e src/rai_sim

# rai_perception (GroundingDINO + Grounded-SAM-2) is required by the
# manipulation demo but isn't a colcon/poetry dependency. Install it here,
# before the torch re-pin below restores the ROCm build it would overwrite.
RUN pip install rai-perception

# Pre-download model weights (~1.5 GB total) so they are cached in the image.
# Without this, the detection and segmentation services download them at
# runtime, taking ~3 minutes and causing launch_robotic_stack to time out
# before the /detection and /segmentation ROS2 services come up.
# The exact URLs match what the services fetch at runtime (see logs):
#   groundingdino_swint_ogc.pth  (661 MB)
#   sam2_hiera_large.pt          (856 MB)
RUN mkdir -p /root/.cache/rai/vision/weights \
    && curl -L -o /root/.cache/rai/vision/weights/groundingdino_swint_ogc.pth \
        https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
    && curl -L -o /root/.cache/rai/vision/weights/sam2_hiera_large.pt \
        https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt

# Reinstall ROCm torch (poetry deps pull CUDA torch).
RUN pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/rocm7.2/ \
        --force-reinstall --no-deps

# The CUDA torch pulled above ships a `triton` that lacks `triton.backends`,
# which the ROCm torch build imports (via inductor) once transformers/timm
# from rai_perception trigger it. Swap it for the matching ROCm triton.
RUN pip uninstall -y triton \
    && pip install pytorch-triton-rocm \
        --index-url https://download.pytorch.org/whl/rocm7.2/ --no-deps

WORKDIR /ryzers

COPY test.sh /ryzers/test_rai.sh
RUN chmod +x /ryzers/test_rai.sh

EXPOSE 8501

CMD ["/ryzers/test_rai.sh"]