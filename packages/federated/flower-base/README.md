# Flower Base Docker Setup

Shared base layer for the Flower federated-learning Ryzers. Installs the
[Flower](https://flower.ai) framework (`flwr[simulation]==1.26.1`) on top
of the default ROCm/PyTorch base image, so the three flower binaries
(`flower-superlink`, `flower-supernode`, `flower-superexec`) and a
ROCm-enabled PyTorch are available to layers that build on this one.

You usually don't build or run this Ryzer on its own — chain it with one of
the role Ryzers:

```sh
ryzers build flower-base flower-superlink   # server box
ryzers build flower-base flower-supernode   # client box
ryzers build flower-base flower-superexec   # serverapp or clientapp runner
```

## Build & Run (standalone smoke test)

```sh
ryzers build flower-base
ryzers run
```

The default `CMD` runs `test_flower-base.sh`, which verifies the CLI
binaries are installed and that `torch.cuda.is_available()` returns true
under ROCm.

## References

- [Flower documentation](https://flower.ai/docs/framework/)
- [Multi-machine Docker tutorial](https://flower.ai/docs/framework/docker/tutorial-deploy-on-multiple-machines.html)

Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
