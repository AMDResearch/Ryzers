# Flower SuperNode - Client Coordinator

The SuperNode coordinates ClientApp execution in Flower federated learning. Each SuperNode manages a partition of the federated clients.

## Build & Run

```bash
# Build the container
ryzers build flower-supernode

# Run SuperNode 1 (partition 0 of 2)
ryzers run flower-supernode --insecure \
  --superlink <superlink-host>:9092 \
  --node-config "partition-id=0 num-partitions=2" \
  --clientappio-api-address 0.0.0.0:9094 \
  --isolation process

# Run SuperNode 2 (partition 1 of 2) - use different port
ryzers run flower-supernode --insecure \
  --superlink <superlink-host>:9092 \
  --node-config "partition-id=1 num-partitions=2" \
  --clientappio-api-address 0.0.0.0:9095 \
  --isolation process
```

## Configuration

| Flag | Description |
|------|-------------|
| `--superlink` | Address of SuperLink Fleet API (port 9092) |
| `--node-config` | Partition configuration for data distribution |
| `--clientappio-api-address` | Address for ClientApp connections |
| `--isolation process` | Run ClientApps in separate processes |
| `--insecure` | Allow unencrypted communication (dev only) |

## Ports

| Port | API | Purpose |
|------|-----|---------|
| 9094+ | ClientAppIO | ClientApp communication |

## Architecture

```
┌─────────────┐
│  SuperLink  │
│    :9092    │
└──────┬──────┘
       │ Fleet API
┌──────▼──────┐
│  SuperNode  │
│    :9094    │
└──────┬──────┘
       │ ClientAppIO API
┌──────▼──────┐
│  SuperExec  │
│ (ClientApp) │
└─────────────┘
```

## References

- [Flower Framework](https://flower.ai/)
- [Flower Docker Tutorial](https://flower.ai/docs/framework/docker/tutorial-quickstart-docker.html)

Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
