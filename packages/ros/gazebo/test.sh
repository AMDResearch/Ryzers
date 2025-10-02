#!/bin/bash

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Test script to verify Gazebo installation - playground sim should open
# https://gazebosim.org/docs/ionic/getstarted/


CMD="gz sim visualize_lidar.sdf"
echo "Running command: $CMD for Gazebo"
$CMD

