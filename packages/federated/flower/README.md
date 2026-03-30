# Flower Federated Learning Ryzer

Demonstrates federated learning using the [Flower framework](https://flower.ai/) with Docker containers.

## Overview

This package implements the [Flower Docker Tutorial](https://flower.ai/docs/framework/docker/tutorial-quickstart-docker.html), demonstrating federated learning with PyTorch across multiple simulated clients.

**Architecture:**
- 1 SuperLink (central coordinator on ports 9091-9093)
- 2 SuperNodes (client managers on ports 9094-9095)
- 3 SuperExec containers (1 ServerApp + 2 ClientApps)
- PyTorch-based federated training on partitioned data

## Prerequisites

- Docker installed and running
- Flower CLI (installed via this ryzer)
- Network ports 9091-9095 available

## Quick Start

```bash
# Navigate to scripts directory
cd packages/workshops/flower/scripts

# One-time setup
./setup_env.sh

# One-time build
./build_containers.sh

# Run the demo
./start_demo.sh
```

## What Happens

1. **Setup** creates:
   - Flower project using `flwr new @flwrlabs/quickstart-pytorch`
   - Docker network for container communication
   - Flower configuration file

2. **Build** creates:
   - Custom SuperExec image with project dependencies

3. **Demo** launches:
   - SuperLink container (central server)
   - 2 SuperNode containers (client managers)
   - 3 SuperExec containers (runtime executors)
   - Federated training across 2 partitioned datasets

## Architecture Details

### Flower Components

- **SuperLink**: Coordinates federated learning rounds, manages global model
- **SuperNodes**: Intermediate layer managing ClientApps on partitioned data
- **SuperExec**: Container runtime executing ServerApp and ClientApps on demand

### Data Flow

1. ServerApp initializes global model in SuperLink
2. SuperLink distributes model to SuperNodes
3. Each SuperNode runs ClientApp on local data partition
4. ClientApps train on local data and return updates
5. SuperLink aggregates updates into new global model
6. Process repeats for configured number of rounds

## Cleanup

```bash
cd packages/workshops/flower/scripts
./cleanup.sh
```

To fully remove the network:
```bash
docker network rm flwr-network
```

## Troubleshooting

**Port already in use:**
- Check if another Flower demo is running: `docker ps`
- Stop conflicting containers: `./cleanup.sh`

**Container communication issues:**
- Verify network exists: `docker network inspect flwr-network`
- Recreate network: `docker network rm flwr-network && docker network create --driver bridge flwr-network`

**Image build failures:**
- Ensure you're in the correct directory
- Check Docker daemon is running: `docker ps`

## References

- [Flower Framework](https://flower.ai/)
- [Flower Docker Tutorial](https://flower.ai/docs/framework/docker/tutorial-quickstart-docker.html)
- [Flower Documentation](https://flower.ai/docs/)
- [Flower GitHub](https://github.com/adap/flower)

Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
