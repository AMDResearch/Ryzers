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
RUN pip install --upgrade pip "setuptools<82" "empy<4" lark && \
    pip install --no-build-isolation openai-whisper==20231117

# RAI framework (pinned commit)
WORKDIR /ryzers
RUN git clone https://github.com/RobotecAI/rai.git

WORKDIR /ryzers/rai
RUN git checkout e54f8ca
RUN vcs import < ros_deps.repos
RUN vcs import < demos.repos

# TODO: Once upstream fixed remove the grabbing_point_tool patch
# sed fixes an upstream bug: GetObjectPositionsTool's forward-ref to
# GetGrabbingPointTool is never rebuilt, so under langchain 1.x / pydantic 2.x the
# model is "not fully defined" and instantiation raises. Any avoids it.
RUN sed -i 's/    get_grabbing_point_tool: "GetGrabbingPointTool"/    get_grabbing_point_tool: Any/' \
        src/rai_core/rai/tools/ros2/manipulation/custom.py && \
    pip install src/rai_core src/rai_whoami

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

# RAI dependencies can pull CUDA torch. Restore the ROCm 7.14 gfx1151 wheels
# from the same repository used by the base image.
# NOTE: --force-reinstall rebuilds the whole dependency tree, not just the three
# wheels named here, so it also drags numpy to 2.x and pillow to 12.x, past the
# base image's numpy<2 pin (which ROS jazzy and rai_core both need). scikit-learn
# 1.3.2, pinned by rai_perception, is a numpy-1-ABI wheel and then dies on import
# with "numpy.dtype size changed" -- which takes the manipulation demo with it.
# The re-pin has to be a separate install: repo.amd.com serves no numpy 1.x, and
# --index-url replaces PyPI rather than adding to it, so constraining the resolve
# above just makes it fail.
RUN pip install --force-reinstall \
    --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
    "torch[device-gfx1151]==2.12.0+rocm7.14.0" \
    "torchvision[device-gfx1151]==0.27.0+rocm7.14.0" \
    "torchaudio==2.11.0+rocm7.14.0" && \
    pip install "numpy<2" "pillow<12"

WORKDIR /ryzers

COPY test.sh /ryzers/test_rai.sh
RUN chmod +x /ryzers/test_rai.sh
COPY manipulation_demo.sh /ryzers/manipulation_demo.sh
RUN chmod +x /ryzers/manipulation_demo.sh
COPY ros_preflight.sh /ryzers/ros_preflight.sh
RUN chmod +x /ryzers/ros_preflight.sh
COPY lemonade_env.sh /ryzers/

EXPOSE 8501
CMD /ryzers/test_rai.sh