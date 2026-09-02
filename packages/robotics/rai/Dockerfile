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

# Protect the base image's ROCm 10 torch stack. rai's Python deps (rai_core,
# rai_bench, rai_sim, rai-perception, openai-whisper) resolve a generic `torch`
# and would otherwise pull CUDA wheels over the baked-in ROCm build. The rocm10
# torch lives only in the base image (no AMD pip index serves it -- whl-multi-arch
# tops out at rocm7.14), so we can't reinstall it; instead we pin it, plus numpy<2
# (ROS jazzy, rai_core, and rai_perception's scikit-learn 1.3.2 numpy-1-ABI wheel
# all need numpy 1.x), via PIP_CONSTRAINT so no later resolve can swap it out.
RUN pip freeze | grep -iE '^(torch|torchvision|torchaudio)==' > /opt/rai-constraints.txt && \
    echo "numpy<2" >> /opt/rai-constraints.txt && \
    echo "rai torch constraints:" && cat /opt/rai-constraints.txt
ENV PIP_CONSTRAINT=/opt/rai-constraints.txt

# Pin empy<4 (required by rosidl_generator_rs). Bump openai-whisper 20231117 ->
# 20250625: the old release pins triton<3, which is unsatisfiable against the
# ROCm 10 torch (2.13.0+rocm10 requires triton==3.8.0+...rocm10). The PIP_CONSTRAINT
# above holds torch on ROCm, so the old whisper's triton<3 now hard-conflicts;
# 20250625 accepts the torch-provided triton 3.8.
RUN pip install --upgrade pip "setuptools<82" "empy<4" lark && \
    pip install --no-build-isolation openai-whisper==20250625

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
# deps, so install from source; rai_perception comes from PyPI.
# rai-perception hard-pins torchvision<0.19.0, unsatisfiable against the ROCm 10
# torchvision 0.28.0 the PIP_CONSTRAINT holds (and rocm10 torchvision isn't on any
# pip index to reinstall, unlike the old rocm7.14 flow). The pin is over-tight --
# only rai-perception itself caps torchvision; its deps (rai-gsam2, rf-groundingdino,
# scikit-learn, transformers) leave torch/torchvision open -- so we install
# rai-perception with --no-deps and pull its real deps explicitly, letting them
# resolve against the base's torchvision 0.28.
RUN pip install src/rai_bench && \
    pip install src/rai_sim && \
    pip install --no-deps rai-perception && \
    pip install "rai-gsam2<2.0.0,>=1.1.2" "rf-groundingdino<0.3.0,>=0.2.0" \
        "scikit-learn<1.4.0,>=1.0.0" "transformers<5.0.0"

# Pre-download detection/segmentation weights (~1.5 GB) via rai-perception's own
# downloader, so model/URL bumps flow through automatically. Lands in the default
# ~/.cache/rai/vision/weights/ the services read at runtime.
# NOTE: relies on the `download-perception-models` console script staying stable.
RUN . /opt/ros/${ROS_DISTRO}/setup.sh && \
    . install/setup.sh && \
    download-perception-models

# Guard: the ROCm 10 torch stack must have survived every rai dep install above
# (the PIP_CONSTRAINT set right after venv activation is what keeps it ROCm).
RUN python -c "import torch; assert torch.version.hip, 'torch is not a ROCm build: '+torch.__version__; print('rai OK torch', torch.__version__, 'hip', torch.version.hip)"

WORKDIR /ryzers

COPY test.sh /ryzers/test_rai.sh
RUN chmod +x /ryzers/test_rai.sh
COPY manipulation_demo.sh /ryzers/manipulation_demo.sh
RUN chmod +x /ryzers/manipulation_demo.sh
COPY lemonade_env.sh /ryzers/

EXPOSE 8501
CMD /ryzers/test_rai.sh