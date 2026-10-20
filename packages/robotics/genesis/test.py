#!/usr/bin/env python3

# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import os
# Use EGL for headless rendering (issue #66)
if 'PYOPENGL_PLATFORM' not in os.environ:
  os.environ['PYOPENGL_PLATFORM'] = 'egl'
  
import genesis as gs

print("Testing Genesis installation...")
print(f"Genesis version: {gs.__version__}")

# Smoke test: one HIP device is visible, Genesis initializes on the AMD GPU backend,
# builds a small scene and steps it. GENESIS_BACKEND=cpu runs the same scene on the CPU
# (the amdgpu backend is the default; gs.vulkan no longer exists in Genesis 1.x).
backend_name = os.environ.get('GENESIS_BACKEND', 'amdgpu')
if backend_name != 'cpu':
    import torch
    assert torch.version.hip, "PyTorch was not built for ROCm"
    n_dev = torch.cuda.device_count()
    assert n_dev >= 1, "No HIP device visible"
    print(f"HIP devices: {n_dev} ({torch.cuda.get_device_name(0)})")
print(f"Initializing Genesis with {backend_name} backend...")
gs.init(backend=getattr(gs, backend_name))

# Headless mode by default (set show_viewer=True if X11 display available)
scene = gs.Scene(show_viewer=False)
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
