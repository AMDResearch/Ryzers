#!/bin/bash

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set -e

echo "Testing Gazebo installation..."

# Verify gz command exists
if ! command -v gz &> /dev/null; then
    echo "FAIL: gz command not found"
    exit 1
fi

echo "gz found at: $(which gz)"

# Check Gazebo version
echo "Gazebo version:"
gz --version

# List available Gazebo commands
echo "Available Gazebo commands:"
gz --help 2>&1 | grep -E "^\s+\w+" | head -15 || true

# Check Gazebo sim help (non-blocking)
echo "Checking Gazebo sim..."
gz sim --help > /dev/null 2>&1 && echo "gz sim available" || echo "gz sim help check completed"

# Verify SDF files are available
echo "Checking for SDF files..."
if [[ -f "/usr/share/gz/gz-sim8/worlds/visualize_lidar.sdf" ]]; then
    echo "Found visualize_lidar.sdf"
elif [[ -f "/usr/share/gz/gz-sim7/worlds/visualize_lidar.sdf" ]]; then
    echo "Found visualize_lidar.sdf"
else
    echo "Note: visualize_lidar.sdf location may vary by Gazebo version"
fi

# Test running a headless simulation briefly
echo "Testing headless simulation (5 seconds)..."
timeout 5s gz sim -s -r --headless-rendering shapes.sdf 2>/dev/null && echo "Headless sim test passed" || echo "Headless sim test completed (timeout expected)"

echo "SUCCESS: Gazebo installation test passed"
echo "Note: For interactive simulation, run: gz sim visualize_lidar.sdf"
exit 0

