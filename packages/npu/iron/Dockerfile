ARG BASE_IMAGE=xdna
FROM ${BASE_IMAGE}

# Where we'll install the build backend
WORKDIR /ryzers
COPY setup.sh /ryzers/setup.sh
COPY entrypoint.sh /ryzers/entrypoint.sh

# Set environment variable to suppress prompts in scripts
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-l", "-c"]

# IRON requires python venv
RUN apt-get -y install python3.12-venv

# Make life easier
RUN apt-get -y install vim

RUN git clone --recursive https://github.com/xilinx/mlir-aie.git /ryzers/mlir-aie && \
    cd /ryzers/mlir-aie && \
    git checkout c105c0b && \
    git submodule update --recursive

# Comment out NPU check - we don't have access to /dev/accel during docker build
RUN sed -i '31s/^/#/' /ryzers/mlir-aie/utils/quick_setup.sh

# Lock down wheels
RUN sed -i '61s|.*|pip install mlir_aie -f https://github.com/Xilinx/mlir-aie/releases/download/latest-wheels/mlir_aie-0.0.1.2025051904+c105c0b-cp312-cp312-manylinux_2_35_x86_64.whl|' /ryzers/mlir-aie/utils/quick_setup.sh
RUN sed -i '64s|.*|pip install llvm_aie -f https://github.com/Xilinx/llvm-aie/releases/download/nightly/llvm_aie-19.0.0.2025041501+b2a279c1-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl|' /ryzers/mlir-aie/utils/quick_setup.sh

# Setup IRON
RUN cd /ryzers/mlir-aie && \
    source /opt/xilinx/xrt/setup.sh && \
    source /ryzers/mlir-aie/utils/quick_setup.sh

# Copy the new AIE pragma utils.h to mlir_aie/include/ so it's picked up when doing #include "aie_kernel_utils.h"
RUN cp /ryzers/mlir-aie/aie_kernels/aie_kernel_utils.h /ryzers/mlir-aie/ironenv/lib/python3.12/site-packages/mlir_aie/include/

# Install jupyter for notebooks
RUN source /ryzers/setup.sh && \
    pip install jupyterlab

# Cleanup
ENV SHELL=/bin/bash

# Make sure to source the setup script on docker start
ENTRYPOINT ["/ryzers/entrypoint.sh"]

COPY test.sh /ryzers/test.sh
CMD ["/ryzers/test.sh"]