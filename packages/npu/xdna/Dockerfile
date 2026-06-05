ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG DRIVER_VERSION="c8d1727"

# required by rocm driver
RUN groupadd -f render && usermod -aG render root

WORKDIR /ryzers

# Suppress prompts in scripts
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
	git \
	curl \
	bash \
	wget

RUN git clone https://github.com/amd/xdna-driver && \
    cd xdna-driver && \
    git checkout $DRIVER_VERSION && \
    git submodule update --init --recursive

# Install build dependencies
RUN cd /ryzers/xdna-driver/tools && \
    apt-get update && \
    ./amdxdna_deps.sh -docker

# Build XRT
RUN cd /ryzers/xdna-driver/xrt/build/ && \
    ./build.sh -npu -opt -noctest

# NPU firmware is sourced exclusively from Ubuntu's linux-firmware deb (see
# extract_npu_firmware.sh below). Replace the driver's download_npufws() with
# an empty stub so the build never fetches .sbin blobs from AMD-hosted
# (repo.radeon.com) URLs. (download_vtd_archives, which fetches the xrt-smi
# validate data, is left intact.)
# Build XDNA driver (-release also generates the .deb package)
RUN cd /ryzers/xdna-driver/build && \
    sed -i '/^download_npufws()/,/^}/c\download_npufws() { :; } # firmware via Ubuntu linux-firmware deb (extract_npu_firmware.sh)' build.sh && \
    ./build.sh -release

# Finally, save and install the XRT/XDNA debian packages
# these will also need to be installed on the host system
# if they haven't been already
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
