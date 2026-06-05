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

# Install build dependencies. amdxdna_deps.sh does not forward -docker to
# xrtdeps.sh, so xrtdeps tries to install linux-headers-$(uname -r) (the HOST
# kernel) and the whole apt batch fails, taking Boost and friends with it.
# Call xrtdeps.sh directly with -docker so kernel headers are skipped, and
# install jq (the only extra thing amdxdna_deps.sh adds) ourselves.
RUN cd /ryzers/xdna-driver && \
    apt-get update && \
    apt-get install -y jq && \
    ./xrt/src/runtime_src/tools/scripts/xrtdeps.sh -docker

# Build XRT
RUN cd /ryzers/xdna-driver/xrt/build/ && \
    ./build.sh -npu -opt -noctest

# Build the XDNA userspace (-release generates the xrt_plugin .deb). We pass
# -nokmod so the in-container build does NOT try to compile the amdxdna kernel
# module against linux-headers-$(uname -r): inside a container uname -r is the
# HOST kernel, so the headers are unavailable on non-Ubuntu hosts (Arch,
# Fedora, custom kernels) and the module cannot be loaded from a container
# anyway. The kernel driver is installed/loaded on the host (see README).
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
