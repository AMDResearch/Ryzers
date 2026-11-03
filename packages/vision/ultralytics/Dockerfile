# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /ryzers 

RUN apt-get update && \
    apt-get install -y libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Install Ultralytics YOLO
RUN pip install --no-cache-dir \
    ultralytics \
    opencv-python \
    pillow \
    "numpy<2" \
    matplotlib

# Download test image
RUN mkdir -p /opt/ultralytics-test-data
RUN wget https://ultralytics.com/images/bus.jpg -O /opt/ultralytics-test-data/bus.jpg
RUN wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt \
    -O /opt/ultralytics-test-data/yolo11n.pt

# Copy test scripts
COPY test_ultralytics.py /ryzers/test_ultralytics.py
COPY demo_ultralytics.py /ryzers/demo_ultralytics.py
CMD ["python3", "/ryzers/test_ultralytics.py"]
