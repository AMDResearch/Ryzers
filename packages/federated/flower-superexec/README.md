# Flower SuperExec - Execution Environment for AMD GPUs

The SuperExec runs ServerApp and ClientApp processes for Flower federated learning. This package includes the PyTorch CIFAR-10 quickstart application optimized for AMD GPUs via ROCm.

## Network Setup

Ensure the Docker bridge network exists (created by flower-superlink):

```bash
docker network create flwr-network  # if not already created
```

## Build & Run

```bash
# Build the container
ryzers build flower-superexec

# Run as ServerApp executor (connects to SuperLink via bridge network)
ryzers run flower-superexec --insecure \
  --executor-type serverapp \
  --executor-config 'superlink="superlink:9091"'

# Run as ClientApp executor (connects to SuperNode via bridge network)
ryzers run flower-superexec --insecure \
  --executor-type clientapp \
  --executor-config 'supernode="supernode-1:9094"'
```

## Included Application

The quickstart application trains a CNN on CIFAR-10:

| Component | Description |
|-----------|-------------|
| `quickstart/task.py` | CNN model, data loading, train/test functions |
| `quickstart/client_app.py` | ClientApp with fit/evaluate |
| `quickstart/server_app.py` | ServerApp with FedAvg strategy |

## AMD GPU Support

ROCm exposes AMD GPUs as CUDA devices. The code automatically detects GPU availability:

```python
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

## Configuration

| Flag | Description |
|------|-------------|
| `--executor-type` | `serverapp` or `clientapp` |
| `--executor-config` | Connection configuration |
| `--insecure` | Allow unencrypted communication (dev only) |

## Full Deployment Example

```bash
# 0. Create bridge network for container DNS resolution
docker network create flwr-network

# 1. Build all containers
ryzers build flower-superlink
ryzers build flower-supernode
ryzers build flower-superexec

# 2. Start SuperLink (named "superlink" on flwr-network)
# Terminal 1:
ryzers run  # runs on ports 9091-9093

# 3. Start SuperNodes (2 partitions)
# Terminal 2: SuperNode 1
docker run --rm -it --network=flwr-network --name=supernode-1 \
  -p 9094:9094 ryzers:flower-supernode \
  flower-supernode --insecure \
  --superlink superlink:9092 \
  --node-config "partition-id=0 num-partitions=2" \
  --clientappio-api-address 0.0.0.0:9094 \
  --isolation process

# Terminal 3: SuperNode 2
docker run --rm -it --network=flwr-network --name=supernode-2 \
  -p 9095:9095 ryzers:flower-supernode \
  flower-supernode --insecure \
  --superlink superlink:9092 \
  --node-config "partition-id=1 num-partitions=2" \
  --clientappio-api-address 0.0.0.0:9095 \
  --isolation process

# 4. Start SuperExec containers
# Terminal 4: ServerApp executor
ryzers run flower-superexec --insecure \
  --executor-type serverapp \
  --executor-config 'superlink="superlink:9091"'

# Terminal 5: ClientApp executor 1
ryzers run flower-superexec --insecure \
  --executor-type clientapp \
  --executor-config 'supernode="supernode-1:9094"'

# Terminal 6: ClientApp executor 2
ryzers run flower-superexec --insecure \
  --executor-type clientapp \
  --executor-config 'supernode="supernode-2:9095"'

# 5. Run the federated learning job
flwr run . local-deployment --stream
```

## Cleanup

```bash
docker network rm flwr-network
```

## References

- [Flower Framework](https://flower.ai/)
- [Flower Docker Tutorial](https://flower.ai/docs/framework/docker/tutorial-quickstart-docker.html)
- [Flower PyTorch Quickstart](https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html)

Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
