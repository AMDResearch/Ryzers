#!/bin/bash

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set -e

echo "Testing ROSCon25 NPU workshop environment..."

# Source environments
source /opt/xilinx/xrt/setup.sh 2>/dev/null || echo "XRT setup not available"
source /opt/ros/${ROS_DISTRO}/setup.sh 2>/dev/null || true

echo "ROS_DISTRO: ${ROS_DISTRO:-not set}"

# Check Python
echo "Python version: $(python3 --version)"

# Check JupyterLab
echo "Checking JupyterLab..."
if ! command -v jupyter-lab &> /dev/null; then
    echo "FAIL: jupyter-lab not found"
    exit 1
fi
echo "JupyterLab version: $(jupyter-lab --version)"

# Check key packages
echo "Checking required packages..."
python3 -c "import matplotlib; print(f'matplotlib: {matplotlib.__version__}')"
python3 -c "import timm; print(f'timm: {timm.__version__}')"

# Check XRT
echo "Checking XRT..."
if command -v xrt-smi &> /dev/null; then
    xrt-smi examine 2>/dev/null | head -10 || echo "xrt-smi examine completed"
else
    echo "xrt-smi not found (may need NPU device)"
fi

# Check notebooks directory
echo "Checking notebooks..."
if [[ -d "/ryzers/notebooks" ]]; then
    echo "Notebooks found: $(find /ryzers/notebooks -name '*.ipynb' 2>/dev/null | wc -l) notebooks"
else
    echo "Warning: notebooks directory not found"
fi

echo "SUCCESS: ROSCon25 NPU workshop test passed"
echo "Note: For JupyterLab, run the entrypoint.sh script"
exit 0
