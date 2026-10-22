#!/usr/bin/env python3

# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import argparse
import os

parser = argparse.ArgumentParser(description="Genesis smoke test (amdgpu backend by default)")
parser.add_argument("--cpu", action="store_true", help="run on the CPU backend instead")
args = parser.parse_args()

# Use EGL for headless rendering (issue #66)
if 'PYOPENGL_PLATFORM' not in os.environ:
  os.environ['PYOPENGL_PLATFORM'] = 'egl'
  
import genesis as gs

print("Testing Genesis installation...")

import genesis as gs
print(f"Genesis version: {gs.__version__}")

# Initialize with the GPU backend (amdgpu on ROCm), or the CPU with --cpu (headless)
print(f"Initializing Genesis with {'CPU' if args.cpu else 'GPU'} backend...")
gs.init(backend=gs.cpu if args.cpu else gs.gpu)
if not args.cpu:
  assert gs.backend == gs.amdgpu, f"expected gs.amdgpu, got {gs.backend}"

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
