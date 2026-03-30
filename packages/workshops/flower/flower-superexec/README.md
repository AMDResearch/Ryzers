# Flower SuperExec Ryzer

SuperExec executes ServerApp and ClientApp logic in Flower federated learning architecture.

## Purpose

The SuperExec component:
- Executes ServerApp (server-side federated logic)
- Executes ClientApp (client-side training logic)
- Uses PyTorch for model training
- Communicates with SuperLink and SuperNodes

## GPU Support

This ryzer supports GPU acceleration for training workloads.

## Usage

This ryzer is typically built and run via the flower workshop scripts:

```bash
cd packages/workshops/flower/scripts
./build_containers.sh
./start_demo.sh
```

Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
