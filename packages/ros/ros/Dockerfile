# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

# Allow build-time argument for ROS distro (jazzy or kilted)
ARG ROS_DISTRO=kilted
ENV ROS_DISTRO=${ROS_DISTRO}

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# 1. Set locale
RUN apt-get update && apt-get install -y locales \
 && locale-gen en_US en_US.UTF-8 \
 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
 && rm -rf /var/lib/apt/lists/*
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# 2. Enable universe repo
RUN apt-get update && apt-get install -y software-properties-common curl gnupg2 \
 && add-apt-repository universe \
 && rm -rf /var/lib/apt/lists/*

# 3. Add ROS 2 apt repository
RUN apt update && sudo apt install curl -y && \
   export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}') && \
   curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb" && \
dpkg -i /tmp/ros2-apt-source.deb


# 4. Install ROS 2 (desktop variant)
RUN apt-get update \
 && apt-get upgrade -y \
 && apt-get install -y ros-${ROS_DISTRO}-desktop \
 && rm -rf /var/lib/apt/lists/*

# 5. (Optional) Install dev tools
RUN apt-get update && apt-get install -y ros-dev-tools \
 && rm -rf /var/lib/apt/lists/*

# 6. Source ROS environment
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc

COPY test.sh /ryzers/test_ros.sh
RUN chmod +x /ryzers/test_ros.sh

# Command to run a test
CMD /ryzers/test_ros.sh  