# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

# Set the OpenCV version as an argument (set in config.yaml)
ARG OPENCV_VERSION

# Debug: python3 is still correctly configured
RUN python3 --version && python3 -m pip --version

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gcc \
    g++ \
    git \
    pkg-config \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    gfortran \
    libopenexr-dev \
    libatlas-base-dev \
    libtbbmalloc2 \
    libtbb-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev \
    libgtk2.0-dev

# Clone OpenCV repositories
WORKDIR /opt
RUN git clone --branch ${OPENCV_VERSION} https://github.com/opencv/opencv.git && \
    git clone --branch ${OPENCV_VERSION} https://github.com/opencv/opencv_contrib.git

# Verify python3 is still correctly configured
RUN python3 --version && python3 -m pip --version

# Build OpenCV
WORKDIR /opt/opencv/build
RUN cmake -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
          -D BUILD_opencv_python3=ON \
          -D BUILD_opencv_python2=OFF \
          -D BUILD_NEW_PYTHON_SUPPORT=ON \
          -D OPENCV_ENABLE_NONFREE=ON .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig

# install Python OpenCV
RUN python3 -m pip install opencv-python

# Clean up
RUN rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /ryzers

# Copy the test script
COPY test.sh /ryzers/opencv_test.sh
RUN chmod +x /ryzers/opencv_test.sh

# Set the entrypoint to the test script
CMD "/ryzers/opencv_test.sh"