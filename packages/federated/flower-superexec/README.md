# Flower SuperExec - Execution Environment for AMD GPUs

The SuperExec runs ServerApp and ClientApp processes for Flower federated learning. This package includes the PyTorch CIFAR-10 quickstart application optimized for AMD GPUs via ROCm.

## Build & Run

```bash
# Build the container
ryzers build flower-superexec

# Run as ServerApp executor (connects to SuperLink)
ryzers run flower-superexec --insecure \
  --executor-type serverapp \
  --executor-config 'superlink="<superlink-host>:9091"'

# Run as ClientApp executor (connects to SuperNode)
ryzers run flower-superexec --insecure \
  --executor-type clientapp \
  --executor-config 'supernode="<supernode-host>:9094"'
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
# 1. Start SuperLink
ryzers build flower-superlink
ryzers run  # runs on ports 9091-9093

# 2. Start SuperNodes (2 partitions)
ryzers build flower-supernode

# Terminal 1: SuperNode 1
ryzers run flower-supernode --insecure \
  --superlink localhost:9092 \
  --node-config "partition-id=0 num-partitions=2" \
  --clientappio-api-address 0.0.0.0:9094 \
  --isolation process

# Terminal 2: SuperNode 2
ryzers run flower-supernode --insecure \
  --superlink localhost:9092 \
  --node-config "partition-id=1 num-partitions=2" \
  --clientappio-api-address 0.0.0.0:9095 \
  --isolation process

# 3. Start SuperExec containers
ryzers build flower-superexec

# Terminal 3: ServerApp executor
ryzers run flower-superexec --insecure \
  --executor-type serverapp \
  --executor-config 'superlink="localhost:9091"'

# Terminal 4: ClientApp executor 1
ryzers run flower-superexec --insecure \
  --executor-type clientapp \
  --executor-config 'supernode="localhost:9094"'

# Terminal 5: ClientApp executor 2
ryzers run flower-superexec --insecure \
  --executor-type clientapp \
  --executor-config 'supernode="localhost:9095"'

# 4. Run the federated learning job
flwr run . local-deployment --stream
```

## References

- [Flower Framework](https://flower.ai/)
- [Flower Docker Tutorial](https://flower.ai/docs/framework/docker/tutorial-quickstart-docker.html)
- [Flower PyTorch Quickstart](https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html)

Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
