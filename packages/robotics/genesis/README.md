###  Genesis Docker Setup

This folder contains the necessary files to build and run a Docker container with Genesis installed using Ryzers.


### Demo (Drone in Interactive Mode)
```sh
ryzers build genesis
ryzers run /ryzers/demo_genesis.sh
```
Found in https://github.com/Genesis-Embodied-AI/Genesis/tree/main/examples/drone

### Tactile Sandbox (Interactive Mode)

A torus, sphere, duck, and dragon over a 20×20 taxel pad, with a live sensor plot.

```bash
ryzers run /ryzers/demo_tactile_sandbox.py
```

**Controls**

| Key | Action |
|---|---|
| Arrow keys | Move object |
| `j` / `k` | Lower / raise |
| `n` / `m` | Rotate |
| `\` | Reset |
| `ESC` | Quit |

**Sensor types:** `kinematic` (default), `depth`, `elastomer`, `proximity`. Select one with `SENSOR`:

```bash
SENSOR=elastomer ryzers run /ryzers/demo_tactile_sandbox.py
```

### Build the Docker Image

```sh
ryzers build genesis
ryzers run
```
### References
- https://genesis-world.readthedocs.io/en/latest/
- https://github.com/Genesis-Embodied-AI/Genesis

Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.