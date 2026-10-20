#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Tactile sensors on a Franka gripper (Genesis 1.4). Backend: GENESIS_BACKEND=amdgpu (default) or cpu.
# Pass --vis (e.g. `ryzers run "/ryzers/demo_genesis.sh --vis"`) to open the viewer on X11.
cd /ryzers/Genesis/examples/sensors
if [ "${GENESIS_BACKEND:-amdgpu}" = "cpu" ]; then
  python tactile_franka.py "$@"
else
  python tactile_franka.py -g "$@"
fi
exit 0
