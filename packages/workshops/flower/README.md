# Flower Federated Learning Ryzer Package

Demonstrates federated learning using the [Flower framework](https://flower.ai/) with custom AMD Ryzer containers.

## Overview

This package implements the [Flower Docker Tutorial](https://flower.ai/docs/framework/docker/tutorial-quickstart-docker.html) using custom Ryzer containers instead of upstream Docker images. It demonstrates federated learning with PyTorch across multiple simulated clients on AMD hardware.

**Architecture:**
- 1 SuperLink ryzer (central coordinator on ports 9091-9093)
- 2 SuperNode ryzers (client managers on ports 9094-9095)
- 3 SuperExec ryzers (1 ServerApp + 2 ClientApps with PyTorch/ROCm)
- PyTorch-based federated training on partitioned data

## Package Structure

```
flower/
├── flower-superlink/     # SuperLink ryzer (coordinator)
├── flower-supernode/     # SuperNode ryzer (client manager)
├── flower-superexec/     # SuperExec ryzer (execution runtime with PyTorch)
├── scripts/              # Orchestration scripts
│   ├── env.sh
│   ├── build_containers.sh
│   ├── setup_env.sh
│   ├── start_demo.sh
│   └── cleanup.sh
└── README.md             # This file
```

## Prerequisites

- Docker installed and running
- Ryzers framework installed
- Network ports 9091-9095 available
- AMD GPU with ROCm support (for GPU acceleration)

## Quick Start

```bash
# Navigate to scripts directory
cd packages/workshops/flower/scripts

# Build all Flower ryzers (run once)
./build_containers.sh

# Setup Flower project (run once)
./setup_env.sh

# Run the demo
./start_demo.sh
```

## What Happens

1. **build_containers.sh** builds three types of ryzers:
   - SuperLink: Flower coordinator (no PyTorch needed)
   - SuperNode: Client manager (no PyTorch needed)
   - SuperExec: Execution runtime (includes PyTorch from base image)

2. **setup_env.sh** creates:
   - Flower project using `flwr new @flwrlabs/quickstart-pytorch`
   - Workspace directory structure
   - Flower configuration file

3. **start_demo.sh** launches:
   - 1 SuperLink container (central server)
   - 2 SuperNode containers (client managers)
   - 3 SuperExec containers (ServerApp + 2 ClientApps)
   - Federated training across 2 partitioned datasets

## Architecture Details

### Flower Components

- **SuperLink**: Coordinates federated learning rounds, manages global model, provides REST API
- **SuperNode**: Intermediate layer managing ClientApps on partitioned data
- **SuperExec**: Container runtime executing ServerApp and ClientApps with PyTorch

### Ryzer Design Principles

- **No Torch Reinstallation**: All ryzers use `--no-deps` when installing `flwr` to avoid reinstalling PyTorch over the AMD-optimized base image
- **Minimal Dependencies**: SuperLink and SuperNode only install necessary gRPC/protobuf dependencies
- **GPU Support**: Only SuperExec has GPU support enabled (for training workloads)
- **Base Image Reuse**: Leverages PyTorch/ROCm from Ryzers base image

### Data Flow

1. ServerApp initializes global model in SuperLink
2. SuperLink distributes model to SuperNodes
3. Each SuperNode runs ClientApp on local data partition via SuperExec
4. ClientApps train on local data using PyTorch and return updates
5. SuperLink aggregates updates into new global model
6. Process repeats for configured number of rounds

## Cleanup

```bash
cd packages/workshops/flower/scripts
./cleanup.sh
```

## Troubleshooting

**Port already in use:**
- Check if containers are running: `docker ps | grep flower`
- Stop conflicting containers: `./cleanup.sh`

**Container communication issues:**
- All containers use `--network host` for localhost communication
- Verify ports 9091-9095 are not blocked by firewall

**PyTorch/ROCm issues:**
- SuperExec uses PyTorch from base image (no reinstallation)
- Check GPU availability: `rocm-smi` or `docker run --rm --device=/dev/kfd --device=/dev/dri flower-superexec:latest python3 -c "import torch; print(torch.cuda.is_available())"`

**Build failures:**
- Ensure you're in the correct directory
- Check base image is available: `docker images | grep ryzer`
- Review build logs for dependency conflicts

## Development Notes

### Adding New Dependencies

When modifying Dockerfiles:

1. **Never reinstall torch** - use `--no-deps` with any package that depends on PyTorch
2. **Test torch availability** - verify PyTorch still works after adding dependencies
3. **Minimize layers** - combine RUN commands where logical

### Customizing the Demo

To use a different Flower project:
- Modify `setup_env.sh` to clone your project instead of using `flwr new`
- Update `start_demo.sh` volume mounts to point to your project
- Ensure your project's `pyproject.toml` dependencies are installed in SuperExec Dockerfile

## References

- [Flower Framework](https://flower.ai/)
- [Flower Docker Tutorial](https://flower.ai/docs/framework/docker/tutorial-quickstart-docker.html)
- [Flower Documentation](https://flower.ai/docs/)
- [Flower GitHub](https://github.com/adap/flower)
- [AMD ROCm](https://www.amd.com/en/products/software/rocm.html)

Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
