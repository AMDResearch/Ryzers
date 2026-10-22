# Genesis Docker Setup

This folder contains the necessary files to build and run a Docker container with Genesis installed using Ryzers.

### Build the Docker Image

```sh
ryzers build genesis
ryzers run
```

Add `--cpu` to run the test on the CPU instead of the GPU: `ryzers run "/ryzers/test_genesis.py --cpu"`.

### Demo

```sh
ryzers run /ryzers/demo_genesis.sh
```

This demo showcases tactile sensors on a Franka gripper. By default it runs headless on the GPU and prints the sensor readings. Options:

- `--vis`: Open the viewer and a live sensor plot (arrow keys and `j`/`k` move the arm, hold `Space` to grasp, `Esc` to quit)
- `--sensor TYPE`: Choose the tactile sensor to simulate: `elastomer` (default), `depth`, `kinematic` or `proximity`
- `--cpu`: Run on the CPU instead of the GPU

For example: `ryzers run "/ryzers/demo_genesis.sh --vis --sensor kinematic"`

### References

- https://genesis-world.readthedocs.io/en/latest/
- https://github.com/Genesis-Embodied-AI/Genesis
- [Tactile Genesis](https://neuroagents-lab.github.io/tactile-genesis/)

Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
