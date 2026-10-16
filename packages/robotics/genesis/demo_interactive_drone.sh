#!/bin/bash

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

sed -i 's/backend=gs\.cpu/backend=gs.gpu/' /ryzers/Genesis/examples/drone/interactive_drone.py
cd Genesis/examples/drone
unset PYOPENGL_PLATFORM  # the viewer needs the window-system OpenGL platform
python interactive_drone.py
exit 0
