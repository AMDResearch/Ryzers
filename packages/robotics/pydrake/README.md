# PyDrake Docker Setup

Drake is a planning, control, and analysis toolbox for nonlinear dynamical systems, developed by MIT and Toyota Research Institute. This package provides PyDrake (Python bindings for Drake) with GPU acceleration via ROCm/PyTorch on AMD Ryzen AI hardware.

## Features

- **MultibodyPlant**: Rigid body dynamics simulation
- **SceneGraph**: Collision detection and visualization
- **Solvers**: Mathematical optimization (SNOPT, IPOPT, Gurobi, Mosek)
- **Meshcat**: Web-based 3D visualization
- **PyTorch Integration**: GPU-accelerated computation via ROCm

## Build & Run

```sh
ryzers build pydrake
ryzers run  # Runs default test
```

### Run the demo (pendulum simulation with visualization)

```sh
ryzers run /ryzers/demo_pydrake.sh
```

Then open http://localhost:7000 to view the Meshcat visualization.

### Run the video demo (falling objects with MP4 output)

```sh
ryzers run /ryzers/demo_video_pydrake.sh
```

Output video saved to `./output/drake_hello_world.mp4`.

### Interactive shell

This package runs in headless mode by default to support SSH. For an interactive shell with TTY, use the `-it` flag:

```sh
ryzers run -it bash
```

## GPU Acceleration

This package includes PyTorch with ROCm backend for GPU-accelerated computation. The `HSA_OVERRIDE_GFX_VERSION=11.0.0` environment variable is set for Strix Point (gfx1150) compatibility.

To verify GPU acceleration:

```python
import torch
print(torch.cuda.is_available())      # Should show True
print(torch.cuda.get_device_name(0))  # Should show AMD Radeon Graphics
```

## References

- [Drake Documentation](https://drake.mit.edu/)
- [Drake GitHub](https://github.com/RobotLocomotion/drake)
- [PyDrake Tutorials](https://deepnote.com/workspace/Drake-0b3b2c53-a7ad-441b-80f8-bf8350752305/project/Tutorials-2b4fc509-aef2-417d-a40d-6071dfed9199)
- [PyTorch ROCm](https://pytorch.org/get-started/locally/)

Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
