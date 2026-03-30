# Flower SuperNode Ryzer

SuperNode manages ClientApp execution in Flower federated learning architecture.

## Purpose

The SuperNode component:
- Manages ClientApp instances on data partitions
- Communicates with SuperLink for coordination
- Provides ClientApp I/O API
- Handles local training execution

## Ports

- **9094**: ClientApp I/O API (first SuperNode)
- **9095**: ClientApp I/O API (second SuperNode)

## Usage

This ryzer is typically built and run via the flower workshop scripts:

```bash
cd packages/workshops/flower/scripts
./build_containers.sh
./start_demo.sh
```

Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
