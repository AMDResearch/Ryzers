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
	wget \
	cmake \
	ninja-build \
	build-essential

RUN git clone https://github.com/amd/xdna-driver && \
    cd xdna-driver && \
    git checkout $DRIVER_VERSION && \
    git submodule update --init --recursive

# Install build dependencies (xrtdeps.sh -docker skips host kernel headers; jq
# is the only extra package amdxdna_deps.sh would add)
RUN cd /ryzers/xdna-driver && \
    apt-get update && \
    apt-get install -y jq && \
    ./xrt/src/runtime_src/tools/scripts/xrtdeps.sh -docker

# Build XRT
RUN cd /ryzers/xdna-driver/xrt/build/ && \
    ./build.sh -npu -opt -noctest

# Build the XDNA userspace (-release generates the xrt_plugin .deb); -nokmod
# skips the kernel module, which is built and loaded on the host (see README)
RUN cd /ryzers/xdna-driver/build && \
    ./build.sh -release -nokmod

# Finally, save and install the XRT/XDNA debian packages
# these will also need to be installed on the host system
# if they haven't been already
RUN mkdir /ryzers/debs && \
    cp /ryzers/xdna-driver/build/Release/xrt_plugin*amd64-amdxdna.deb /ryzers/debs/ && \
    cp /ryzers/xdna-driver/xrt/build/Release/xrt*-amd64-base.deb /ryzers/debs/ && \
	dpkg -i /ryzers/debs/*.deb

# Cleanup
ENV SHELL=/bin/bash

COPY test.sh /ryzers/
CMD ["/ryzers/test.sh"]
