#!/usr/bin/env python3

# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'

print("Testing Genesis installation...")

import genesis as gs
print(f"Genesis version: {gs.__version__}")

# Initialize with Vulkan backend (headless)
print("Initializing Genesis with Vulkan backend...")
gs.init(backend=gs.vulkan)

# Create scene WITHOUT viewer (headless mode)
print("Creating headless scene...")
scene = gs.Scene(show_viewer=False)

# Add entities
print("Adding entities...")
plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)

# Build scene
print("Building scene...")
scene.build()

# Run simulation steps
print("Running 100 simulation steps...")
for i in range(100):
    scene.step()
    if (i + 1) % 25 == 0:
        print(f"  Step {i + 1}/100 completed")

print("SUCCESS: Genesis simulation test passed")
print("Note: For interactive viewer, use: scene = gs.Scene(show_viewer=True)")
