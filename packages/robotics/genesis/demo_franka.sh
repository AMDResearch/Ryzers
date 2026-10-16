#!/bin/bash

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

unset PYOPENGL_PLATFORM  # the viewer needs the window-system OpenGL platform
python3 /ryzers/demo_genesis.py
exit 0
