# Flower
Flower is a framework for federated learning. This Ryzer demonstrates how to use Flower on AMD GPUs. This example is based on the Flower example found [here](https://flower.ai/docs/framework/docker/tutorial-quickstart-docker.html).

## Quick Start

The Flower ryzer comes pre-configured with the PyTorch quickstart project and will automatically run `flwr run . local-deployment --stream` when launched.

### Build the Flower ryzer
```bash
ryzers build flower
```

### Run the quickstart
```bash
ryzers run
```

This will launch the Flower quickstart project, which demonstrates federated learning with PyTorch on AMD GPUs.

## Advanced Usage

### Build individual containers
Unlike other Ryzers, Flower can use several containers for distributed deployment. From the Ryzers directory, you can build:

```bash
ryzers build flower superexec --name superexec
ryzers build flower superlink --name superlink
ryzers build flower supernode --name supernode
```

### Create the network
Flower communicates between containers using a Docker network. Create a dedicated network:
```bash
docker network create --driver bridge flwr-network
```

### Run distributed components
Use the scripts in `packages/federated/flower/scripts/` to launch the distributed architecture:
- `start_superlink.sh` - Launch the federation coordinator
- `start_supernodes.sh` - Launch client nodes (2 instances)
- `start_superexecs.sh` - Launch execution containers

Or run all components:
```bash
./packages/federated/flower/scripts/start_demos.sh
```

## Configuration

The Flower configuration is automatically set up in `~/.flwr/config.toml`:
```toml
[superlink.local-deployment]
address = "127.0.0.1:9093"
insecure = true
```

You can verify the configuration with:
```bash
flwr config list
```

## Notes

- The quickstart project dependencies (torch/torchvision) are automatically removed from `pyproject.toml` since they're already installed in the base Ryzer environment
- The example demonstrates federated learning across 2 clients using AMD GPU acceleration
- Based on the official Flower quickstart tutorial: https://flower.ai/docs/framework/docker/tutorial-quickstart-docker.html
