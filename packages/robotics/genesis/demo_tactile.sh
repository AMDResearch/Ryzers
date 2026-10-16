#!/bin/bash

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Tactile Genesis: https://neuroagents-lab.github.io/tactile-genesis/
# coacd must be imported before torch: the ROCm libraries torch preloads export their own
# spdlog, which crashes convex decomposition otherwise.
unset PYOPENGL_PLATFORM  # the viewer needs the window-system OpenGL platform
python3 -c 'import coacd, runpy; runpy.run_path("/ryzers/Genesis/examples/sensors/tactile_sandbox.py", run_name="__main__")' --vis --gpu --sensor "${1:-kinematic}"
exit 0
