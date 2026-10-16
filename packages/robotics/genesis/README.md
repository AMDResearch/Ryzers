### Genesis Docker Setup

This folder contains the necessary files to build and run a Docker container with Genesis installed using Ryzers.

### Build the Docker Image

```sh
ryzers build genesis
ryzers run
```

### Demos

| Demo | Command | What it shows |
|---|---|---|
| SPH liquid | `ryzers run /ryzers/demo_genesis.sh` | A viscous liquid released above the ground |
| Drone | `ryzers run /ryzers/demo_interactive_drone.sh` | A quadrotor flown from the keyboard |
| Franka IK | `ryzers run /ryzers/demo_franka.sh` | Nine Franka arms tracking IK targets (Ctrl+C to stop) |
| Tactile sandbox | `ryzers run "/ryzers/demo_tactile.sh kinematic"` | Objects pressed on a taxel pad with a live sensor plot |

Tactile sensor options: `kinematic` (default), `depth`, `contact`, `elastomer`, `proximity`.

### References

- https://genesis-world.readthedocs.io/en/latest/
- https://github.com/Genesis-Embodied-AI/Genesis
- [Drone Examples](https://github.com/Genesis-Embodied-AI/Genesis/tree/main/examples/drone)
- [Tactile Genesis](https://neuroagents-lab.github.io/tactile-genesis/)

Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
