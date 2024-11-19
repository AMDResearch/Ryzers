#!/bin/bash

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Test script to verify Gazebo installation - playground sim should open
# https://gazebosim.org/docs/ionic/getstarted/


# Check if GAZEBO_VERSION is set
if [ -z "$GAZEBO_VERSION" ]; then
  echo "GAZEBO_VERSION is not set. Please set the GAZEBO_VERSION variable."
  exit 1
fi

# Check the value of GAZEBO_VERSION
if [ "$GAZEBO_VERSION" == "fortress" ]; then
  CMD="ign gazebo sim visualize_lidar.sdf"
else
  CMD="gz sim visualize_lidar.sdf"
fi

# Echo the command
echo "Running command: $CMD for Gazebo: $GAZEBO_VERSION"

# Execute the command
$CMD

