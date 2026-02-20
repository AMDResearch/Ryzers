#!/bin/bash
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set -e

cd /ryzers/rai
source install/setup.bash
source /opt/ros/${ROS_DISTRO}/setup.sh
streamlit run examples/manipulation-demo-streamlit.py
