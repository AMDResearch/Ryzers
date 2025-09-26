#!/usr/bin/env python3

# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'
 
import genesis as gs

gs.init(backend=gs.vulkan)

scene = gs.Scene(show_viewer=True)
plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)
scene.build()

for i in range(1000):
    scene.step()
