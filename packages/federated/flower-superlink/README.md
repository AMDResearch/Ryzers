# Flower SuperLink - Federation Coordinator

The SuperLink is the central coordination server for Flower federated learning. It manages communication between ServerApps and SuperNodes.

## Build & Run

```bash
# Build the container
ryzers build flower-superlink

# Run the SuperLink
ryzers run

# Or with explicit command
ryzers run flower-superlink --insecure --isolation process
```

## Ports

| Port | API | Purpose |
|------|-----|---------|
| 9091 | ServerAppIO | Receives ServerApp requests |
| 9092 | Fleet | SuperNode coordination |
| 9093 | Control | Administrative commands |

## Architecture

```
                    ┌─────────────┐
                    │  SuperLink  │
                    │  :9091-9093 │
                    └──────┬──────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
     ┌──────▼─────┐ ┌──────▼─────┐ ┌──────▼─────┐
     │ SuperNode  │ │ SuperNode  │ │ SuperExec  │
     │     1      │ │     2      │ │ (ServerApp)│
     └──────┬─────┘ └──────┬─────┘ └────────────┘
            │              │
     ┌──────▼─────┐ ┌──────▼─────┐
     │ SuperExec  │ │ SuperExec  │
     │(ClientApp) │ │(ClientApp) │
     └────────────┘ └────────────┘
```

## Usage with Other Components

1. Start SuperLink first
2. Start SuperNodes connecting to SuperLink:9092
3. Start SuperExec for ServerApp connecting to SuperLink:9091
4. Start SuperExec for ClientApps connecting to SuperNodes

## References

- [Flower Framework](https://flower.ai/)
- [Flower Docker Tutorial](https://flower.ai/docs/framework/docker/tutorial-quickstart-docker.html)

Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
