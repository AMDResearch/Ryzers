# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /ryzers

# EGL/Mesa dependencies for headless VTK rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    libegl1-mesa-dev libgles2-mesa-dev \
 && rm -rf /var/lib/apt/lists/*

# Drake and dependencies
RUN pip3 install --no-cache-dir --break-system-packages \
    "drake>=1.30.0" \
    "meshcat>=0.3.2" \
    "opencv-python-headless>=4.8.0"

# Copy test and demo scripts
COPY test.py /ryzers/test_pydrake.py
COPY demo.py /ryzers/demo_pydrake.py
RUN chmod +x /ryzers/test_pydrake.py /ryzers/demo_pydrake.py

EXPOSE 7000

CMD /ryzers/test_pydrake.py
