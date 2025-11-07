#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

source env.sh

set_video_format() {
    local DEVICE="$1"

    if [ -z "$DEVICE" ]; then
        echo "Usage: set_video_format /dev/videoX"
        return 1
    fi

    if [ -e "$DEVICE" ]; then
        echo "$DEVICE found. Setting video format to 640x480..."
        v4l2-ctl -d "$DEVICE" --set-fmt-video=width=640,height=480
    else
        echo "$DEVICE not found.  Please plug in 2 webcams for this demo and verify $DEVICE exists."
        return 1
    fi
}

# Create virtual environment and clone a fixed version of Ryzers repo
pushd $SPSDEMO_PATH
    python3 -m venv $SPSDEMO_VENV
    source $SPSDEMO_VENV/bin/activate

    git clone ../../../ $SPSDEMO_RYZERS 
    cp patches/smolvm_demo.sh $SPSDEMO_RYZERS/packages/vlm/smolvlm/demo.sh
    cp patches/ryzenai_demo_ros.sh $SPSDEMO_RYZERS/packages/npu/ryzenai_cvml/demo_ros.sh

    pushd $SPSDEMO_RYZERS
        pip install .
    popd
popd

# Setup xdna driver (only needed one time)
# For NPU to work secure boot needs to be disabled in BIOS
pushd $SPSDEMO_RYZERS
    ./packages/npu/xdna/install_xdna.sh
popd

# Set video format for two webcams
set_video_format /dev/video0
set_video_format /dev/video2
