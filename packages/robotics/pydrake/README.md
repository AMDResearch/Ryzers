# PyDrake Docker Setup

Drake is a planning, control, and analysis toolbox for nonlinear dynamical systems, developed by the Toyota Research Institute. This package provides PyDrake (Python bindings for Drake) with GPU acceleration via ROCm/JAX on AMD Ryzen AI hardware.

## Features

- **MultibodyPlant**: Rigid body dynamics simulation
- **SceneGraph**: Collision detection and visualization
- **Solvers**: Mathematical optimization (SNOPT, IPOPT, Gurobi, Mosek)
- **Meshcat**: Web-based 3D visualization
- **PyTorch Integration**: GPU-accelerated computation via ROCm

## Build & Run

```sh
ryzers build pydrake
ryzers run
```

### Run the demo (pendulum simulation with visualization)

```sh
ryzers run python3 /ryzers/demo_pydrake.py
```

Then open http://localhost:7000 to view the Meshcat visualization.

### Interactive shell

```sh
ryzers run bash
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
