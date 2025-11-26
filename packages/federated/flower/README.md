### Flower

This directory contains the docker configuration to run the federated learning framework Flower.

### Build and run the Docker Image

To build and run a Docker container with Flower, run:

```sh
ryzers build flower
ryzers run
```

### Demo

Inside the container, there is a test application running on Pytorch with ROCm backend. To run it, cd into the testapp folder and run `flwr run .`.


Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
