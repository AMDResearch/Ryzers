# Flower SuperLink Ryzer

SuperLink is the central coordinator in Flower federated learning architecture.

## Purpose

The SuperLink component:
- Coordinates federated learning rounds
- Manages the global model state
- Communicates with SuperNodes and SuperExec containers
- Provides REST API for monitoring

## Ports

- **9091**: SuperExec communication
- **9092**: SuperNode communication
- **9093**: REST API

## Usage

This ryzer is typically built and run via the flower workshop scripts:

```bash
cd packages/workshops/flower/scripts
./build_containers.sh
./start_demo.sh
```

Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
