# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /ryzers 

# Install Ultralytics YOLO
RUN pip install --no-cache-dir \
    ultralytics \
    opencv-python \
    pillow \
    "numpy<2" \
    matplotlib

# Download test image
RUN mkdir -p /ryzers/data_ultralytics/
RUN wget https://ultralytics.com/images/bus.jpg -O /ryzers/data_ultralytics/bus.jpg

# Copy test scripts
COPY test_ultralytics.py /ryzers/test_ultralytics.py
COPY demo_ultralytics.py /ryzers/demo_ultralytics.py
CMD ["python3", "/ryzers/test_ultralytics.py"]
