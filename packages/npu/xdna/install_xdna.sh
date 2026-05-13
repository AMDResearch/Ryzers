#!/bin/bash
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set -e

DRIVER_VERSION=854ff04
USE_RYZER=true
ROOT_DIR=$(pwd)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check kernel version
kernel_version=$(uname -r | cut -d'-' -f1)
if [[ "$kernel_version" < "6.10" ]]; then
    echo "Current kernel version ${kernel_version} not supported. Required kernel >=6.10."
    exit 1
fi

# Detect distro
if [ -f /etc/os-release ]; then
    . /etc/os-release
    DISTRO_ID="$ID"
else
    DISTRO_ID="unknown"
fi

install_firmware() {
    if [ -f "$SCRIPT_DIR/extract_npu_firmware.sh" ]; then
        echo "Extracting NPU firmware..."
        sudo bash "$SCRIPT_DIR/extract_npu_firmware.sh" /lib/firmware/amdnpu
    fi
}

reload_driver() {
    echo "Reloading amdxdna driver..."
    sudo modprobe -r amdxdna 2>/dev/null || true
    sudo modprobe amdxdna
}

# --- Debian / Ubuntu ---
install_ubuntu() {
    ubuntu_ver=$(lsb_release -rs 2>/dev/null || echo "unknown")
    case "$ubuntu_ver" in
        24.04|24.10) ;;
        *) echo "Warning: Ubuntu $ubuntu_ver not officially tested (24.04/24.10 supported)." ;;
    esac

    if dpkg-query -W xrt_plugin-amdxdna 2>/dev/null; then
        echo "xrt_plugin-amdxdna is already installed."
        install_firmware
        reload_driver
        return
    fi

    echo "Installing build dependencies..."
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
            git clone https://github.com/amd/xdna-driver
        fi
        cd xdna-driver && git checkout $DRIVER_VERSION && git submodule update --init --recursive
        source ./tools/amdxdna_deps.sh
        cd xrt/build/ && ./build.sh -npu -opt
        cd ../../build && ./build.sh -release && ./build.sh -package
        cp $ROOT_DIR/xdna-driver/xrt/build/Release/xrt_*-amd64-base.deb $ROOT_DIR/
        cp $ROOT_DIR/xdna-driver/build/Release/xrt_plugin*-amdxdna.deb $ROOT_DIR/
        cd $ROOT_DIR
    fi

    echo "Installing debian packages..."
    sudo dpkg -i ./xrt_*-amd64-base.deb || sudo apt -y install --fix-broken
    sudo dpkg -i ./xrt_plugin*-amdxdna.deb || sudo apt -y install --fix-broken

    install_firmware
    reload_driver
}

# --- Arch Linux ---
install_arch() {
    echo "Arch Linux detected."
    # Kernel 6.12+ has amdxdna built-in; just need firmware
    if lsmod | grep -q amdxdna; then
        echo "amdxdna kernel module already loaded."
    else
        echo "Loading amdxdna module..."
        sudo modprobe amdxdna || echo "Warning: amdxdna module not available in kernel $(uname -r)"
    fi
    install_firmware
    if lsmod | grep -q amdxdna; then
        reload_driver
    fi
}

# --- Dispatch ---
case "$DISTRO_ID" in
    ubuntu|debian)
        install_ubuntu
        ;;
    arch|endeavouros|manjaro)
        install_arch
        ;;
    *)
        echo "Unsupported distro: $DISTRO_ID"
        echo "Attempting firmware-only install (assumes kernel has amdxdna built-in)..."
        install_firmware
        reload_driver
        ;;
esac

echo "=== XDNA setup complete ==="
