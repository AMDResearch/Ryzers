#!/usr/bin/env python3

# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import os
# Use EGL for headless rendering (issue #66)
if 'PYOPENGL_PLATFORM' not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

import genesis as gs

gs.init(backend=gs.vulkan)

# Headless mode by default (set show_viewer=True if X11 display available)
scene = gs.Scene(show_viewer=False)
plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)
scene.build()

for i in range(1000):
    scene.step()
