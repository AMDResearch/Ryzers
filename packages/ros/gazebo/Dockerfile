# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

ARG GAZEBO_VERSION

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
    curl lsb-release gnupg

RUN curl https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
RUN apt-get update
RUN apt-get install -y gz-${GAZEBO_VERSION}


COPY test.sh /ryzers/test_gazebo.sh
RUN chmod +x /ryzers/test_gazebo.sh

# Command to run a test
CMD /ryzers/test_gazebo.sh  