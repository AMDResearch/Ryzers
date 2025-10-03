#!/bin/bash

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Test script to verify ROS installation
source /opt/ros/${ROS_DISTRO}/setup.bash
ros2 run turtlesim turtlesim_node
