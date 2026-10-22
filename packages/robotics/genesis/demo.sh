#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Genesis demo: tactile sensors on a Franka Panda gripper
# (upstream examples/sensors/tactile_franka.py).
#
# Defaults to the amdgpu backend, headless (a scripted grasp that prints sensor readings).
#   --cpu          run on the CPU backend instead
#   --vis          open the viewer and live sensor plot (needs an X11 display)
#   --sensor TYPE  elastomer (default), depth, kinematic or proximity

backend_args=(--gpu)   # upstream defaults to CPU; this demo defaults to amdgpu
passthrough_args=()

for arg in "$@"; do
    case "$arg" in
        --cpu)
            backend_args=()
            ;;
        --vis)
            unset PYOPENGL_PLATFORM   # config.yaml sets egl for headless; the viewer needs X11
            passthrough_args+=("$arg")
            ;;
        *)
            passthrough_args+=("$arg")   # --sensor and other upstream options
            ;;
    esac
done

cd /ryzers/Genesis/examples/sensors
exec python3 tactile_franka.py "${backend_args[@]}" "${passthrough_args[@]}"
