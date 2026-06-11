#!/bin/bash

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set -e

echo "Testing ROS 2 installation..."

# Source ROS environment
source /opt/ros/${ROS_DISTRO}/setup.bash

echo "ROS_DISTRO: ${ROS_DISTRO}"
echo "ROS_VERSION: ${ROS_VERSION}"

# Verify ros2 command exists
if ! command -v ros2 &> /dev/null; then
    echo "FAIL: ros2 command not found"
    exit 1
fi

echo "ros2 found at: $(which ros2)"

# Check ROS 2 daemon
echo "Checking ROS 2 daemon..."
ros2 daemon status || ros2 daemon start || true

# List available packages
echo "Listing installed ROS 2 packages..."
ros2 pkg list | head -20
echo "... (truncated, $(ros2 pkg list | wc -l) total packages)"

# Verify turtlesim is installed
echo "Checking turtlesim package..."
if ros2 pkg list | grep -q turtlesim; then
    echo "turtlesim package found"
    ros2 pkg executables turtlesim
else
    echo "Warning: turtlesim not installed"
fi

# Test basic ROS 2 functionality
echo "Testing ros2 topic list..."
timeout 5s ros2 topic list 2>/dev/null || echo "No topics (expected without running nodes)"

echo "Testing ros2 node list..."
timeout 5s ros2 node list 2>/dev/null || echo "No nodes (expected)"

echo "SUCCESS: ROS 2 installation test passed"
echo "Note: For interactive turtlesim, run: ros2 run turtlesim turtlesim_node"
exit 0
