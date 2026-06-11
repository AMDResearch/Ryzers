#!/bin/bash

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set -e

echo "Testing ROSCon25 GPU workshop environment..."

# Source ROS environment
source /opt/ros/${ROS_DISTRO}/setup.bash
source /ryzers/notebooks/vlm_ros/install/setup.bash 2>/dev/null || true

echo "ROS_DISTRO: ${ROS_DISTRO}"

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
python3 -c "import torch; print(f'torch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import numpy; print(f'numpy: {numpy.__version__}')"
python3 -c "import matplotlib; print(f'matplotlib: {matplotlib.__version__}')"

# Check ROS 2
echo "Checking ROS 2..."
ros2 pkg list | head -5
echo "... ($(ros2 pkg list | wc -l) total packages)"

# Check Vulkan
echo "Checking Vulkan..."
if command -v vulkaninfo &> /dev/null; then
    vulkaninfo --summary 2>/dev/null | head -5 || echo "Vulkan info not available"
else
    echo "vulkaninfo not found"
fi

# Check notebooks directory
echo "Checking notebooks..."
if [[ -d "/ryzers/notebooks" ]]; then
    echo "Notebooks found: $(find /ryzers/notebooks -name '*.ipynb' | wc -l) notebooks"
else
    echo "Warning: notebooks directory not found"
fi

echo "SUCCESS: ROSCon25 GPU workshop test passed"
echo "Note: For JupyterLab, run the entrypoint.sh script"
exit 0
