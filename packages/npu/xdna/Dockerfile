ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG DRIVER_VERSION="854ff04"

RUN groupadd -f render && usermod -aG render root

WORKDIR /ryzers

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
	git \
	curl \
	bash \
	wget \
	cmake \
	build-essential \
	libboost-all-dev \
	pkg-config

RUN git clone https://github.com/amd/xdna-driver && \
    cd xdna-driver && \
    git checkout $DRIVER_VERSION && \
    git submodule update --init --recursive

RUN cd /ryzers/xdna-driver/tools && \
    apt-get update && \
    ./amdxdna_deps.sh -docker

# Build XRT
RUN cd /ryzers/xdna-driver/xrt/build/ && \
    ./build.sh -npu -opt -noctest

# Build XDNA driver
RUN cd /ryzers/xdna-driver/build && \
    ./build.sh -release && \
    ./build.sh -package

# Install XRT/XDNA debs (also needed on host via install_xdna.sh)
RUN mkdir /ryzers/debs && \
    cp /ryzers/xdna-driver/build/Release/xrt_plugin*amd64-amdxdna.deb /ryzers/debs/ && \
    cp /ryzers/xdna-driver/xrt/build/Release/xrt*-amd64-base.deb /ryzers/debs/ && \
	dpkg -i /ryzers/debs/*.deb

# Extract NPU firmware from Ubuntu's linux-firmware deb
COPY extract_npu_firmware.sh /ryzers/extract_npu_firmware.sh
RUN chmod +x /ryzers/extract_npu_firmware.sh && \
    /ryzers/extract_npu_firmware.sh

ENV SHELL=/bin/bash

COPY test.sh /ryzers/
CMD ["/ryzers/test.sh"]
