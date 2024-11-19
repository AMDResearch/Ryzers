# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Set the environment variable required for HIP kernel compilation
ENV HSA_OVERRIDE_GFX_VERSION=11.0.0
    
WORKDIR /ryzers

# Clone the repository and install segment-anything
RUN git clone https://github.com/facebookresearch/segment-anything.git && \
    cd segment-anything && \
    git checkout dca509fe793f601edb92606367a655c15ac00fdf && \
    sed -i 's|, "onnx.*"||' setup.py && \
    sed -i 's|, "torchvision[^"]*"||g' setup.py && \
    sed -i 's|, "torch[^"]*"||g' setup.py && \
    pip3 install .

RUN apt-get install -y libjpeg-dev python3-dev python3-pip 

RUN pip3 install "numpy<2" matplotlib pycocotools tqdm pillow requests onnxruntime jupyterlab opencv-python


COPY test_sam.py /ryzers/
COPY test.sh /ryzers/test_sam.sh
RUN chmod +x /ryzers/test_sam.sh

# Replace device "cuda" with "cpu" in all .ipynb files under the original notebooks folder
RUN find /ryzers/segment-anything/notebooks -type f -name "*.ipynb" \
    -exec sed -i 's/device = \\"cuda\\"/device = \\"cpu\\"/g' {} +

RUN mkdir -p /ryzers/notebooks && \
    mv /ryzers/segment-anything/notebooks /ryzers/notebooks/segment-anything

WORKDIR /ryzers/notebooks

CMD /ryzers/test_sam.sh