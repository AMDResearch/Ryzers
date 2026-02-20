# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install additional dependencies for RAI
RUN apt-get update && apt-get install -y \
    git \
    unzip \
    ffmpeg \
    libportaudio2 \
    python3-venv \
    ros-${ROS_DISTRO}-rai-interfaces \
    ros-${ROS_DISTRO}-moveit \
    ros-${ROS_DISTRO}-gazebo-msgs \
    ros-${ROS_DISTRO}-moveit-resources-panda-moveit-config \
    ros-${ROS_DISTRO}-depth-image-proc \
    ros-${ROS_DISTRO}-ackermann-msgs \
    ros-${ROS_DISTRO}-sdformat-urdf \
 && rm -rf /var/lib/apt/lists/*

# Configure linker to find sdformat vendor libraries
RUN echo "/opt/ros/${ROS_DISTRO}/opt/sdformat_vendor/lib" > /etc/ld.so.conf.d/ros-sdformat.conf && \
    ldconfig

# Use existing virtual env from base image
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN echo "source $VIRTUAL_ENV/bin/activate" >> /root/.bashrc

# Install Poetry into the venv
RUN pip install --upgrade pip "setuptools<82" poetry empy lark && \
    pip install --no-build-isolation openai-whisper==20231117

# Clone and install RAI framework
WORKDIR /ryzers
RUN git clone https://github.com/RobotecAI/rai.git

WORKDIR /ryzers/rai
RUN vcs import < ros_deps.repos
RUN vcs import < demos.repos

RUN poetry config virtualenvs.create false && \
    poetry install --all-groups

RUN rosdep init && \
    rosdep update && \
    rosdep install --from-paths src/examples/rai-manipulation-demo/ros2_ws/src --ignore-src -r -y

# Download demo
RUN . ./scripts/download_demo.sh manipulation

# Build ROS 2 packages (need to source ROS environment first)
RUN . /opt/ros/${ROS_DISTRO}/setup.sh && \
    colcon build --symlink-install

# Copy test scripts
WORKDIR /ryzers
COPY test.sh /ryzers/test_rai.sh
RUN chmod +x /ryzers/test_rai.sh

COPY manipulation_demo.sh /ryzers/manipulation_demo.sh
RUN chmod +x /ryzers/manipulation_demo.sh

# Expose common ports (Streamlit frontend, LangFuse)
EXPOSE 8501

# Default command runs test
CMD /ryzers/test_rai.sh
