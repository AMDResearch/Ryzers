#!/bin/bash

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set -e

DRIVER_VERSION=51b9400
USE_RYZER=true
ROOT_DIR=$(pwd)
CURRENT_USER=$(whoami)

# Check Ubuntu version
ubuntu_ver=$(lsb_release -rs | awk '{print $1}')

if [ "$ubuntu_ver" == "24.04" ]; then
    BASE_IMAGE="ubuntu:24.04"
elif [ "$ubuntu_ver" == "24.10" ]; then
    BASE_IMAGE="ubuntu:24.10"
else
    echo "Not supported Ubuntu release: $ubuntu_ver"
    echo "Only Ubuntu 24.04.2 and Ubuntu 24.10 are supported."
    exit 1
fi

# Check kernel version
kernel_version=$(uname -r | cut -d'-' -f1)

if [[ "$kernel_version" < "6.10" ]]; then
    echo "Current kernel version ${kernel_version} not supported. Required kernel >6.10."
    exit 1
fi

# Host machine setup
if dpkg-query -W xrt_plugin-amdxdna; then
    echo "xrt_plugin-amdxdna is already installed, skipping host installs..."
else
    echo "Installing dependencies"
    sudo apt-get install -y build-essential gcc-x86-64-linux-gnu libgl-dev libxdmcp-dev \
	    bzip2 libalgorithm-diff-perl libglx-dev lto-disabled-list dkms libalgorithm-diff-xs-perl \
	    libhwasan0 make dpkg-dev libalgorithm-merge-perl libitm1 ocl-icd-opencl-dev fakeroot libasan8 \
	    liblsan0 opencl-c-headers g++ libboost-filesystem1.83.0 libquadmath0 opencl-clhpp-headers g++-14 \
	    libboost-program-options1.83.0 libstdc++-14-dev uuid-dev g++-14-x86-64-linux-gnu libcc1-0 libtsan2 \
	    x11proto-dev g++-x86-64-linux-gnu libdpkg-perl libubsan1 xorg-sgml-doctools gcc libfakeroot \
	    libx11-dev xtrans-dev gcc-14 libfile-fcntllock-perl libxau-dev gcc-14-x86-64-linux-gnu \
	    libgcc-14-dev libxcb1-dev
    
    if $USE_RYZER ; then
        ryzers build xdna --name xdna
        docker run --rm -v $(pwd):/host_dir xdna:latest bash -c "cp -v /ryzers/debs/*.deb /host_dir/"
    else
        if [ ! -d "xdna-driver" ]; then
            echo "Cloning repository..."
            git clone https://github.com/amd/xdna-driver
        else
            echo "Repository already exists, skipping clone..."
        fi

        cd xdna-driver
        git checkout $DRIVER_VERSION
        git submodule update --init --recursive

        source ./tools/amdxdna_deps.sh

        cd xrt/build/
        ./build.sh -npu -opt

        cd ../../build
        ./build.sh -release
        ./build.sh -package

        # copy generated debs
        cp $ROOT_DIR/xdna-driver/xrt/build/Release/xrt_*-amd64-base.deb $ROOT_DIR/
        cp $ROOT_DIR/xdna-driver/build/Release/xrt_plugin*-amdxdna.deb $ROOT_DIR/

        cd $ROOT_DIR
    fi
    
    echo "Installing debian packages..."
    if ! sudo dpkg -i ./xrt_*-amd64-base.deb; then
        sudo apt -y install --fix-broken
    fi

    if ! sudo dpkg -i ./xrt_plugin*-amdxdna.deb; then
        sudo apt -y install --fix-broken
    fi
fi
