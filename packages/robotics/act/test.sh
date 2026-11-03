#!/bin/bash

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

cd act
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    python3 record_sim_episodes.py --task_name sim_transfer_cube_scripted --dataset_dir data --num_episodes 1
