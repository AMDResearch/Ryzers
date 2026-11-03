# Flower Base Docker Setup

Shared base layer for the Flower federated-learning Ryzers. Installs the
[Flower](https://flower.ai) framework (`flwr[simulation]==1.26.1`) on top
of the default ROCm/PyTorch base image, so the three flower binaries
(`flower-superlink`, `flower-supernode`, `flower-superexec`) and a
ROCm-enabled PyTorch are available to layers that build on this one.

You usually don't build or run this Ryzer on its own — chain it with one of
the role Ryzers:

```sh
# --name sets the final image tag (it defaults to "ryzerdocker", not the
# last package name), so each role gets its own image + run-script.
ryzers build --name flower-superlink flower-base flower-superlink   # server box
ryzers build --name flower-supernode flower-base flower-supernode   # client box
ryzers build --name flower-superexec flower-base flower-superexec   # serverapp or clientapp runner
```

## Build & Run (standalone smoke test)

```sh
ryzers build flower-base
ryzers run
```

The default `CMD` runs `test_flower-base.sh`, which verifies the CLI
binaries are installed and that `torch.cuda.is_available()` returns true
under ROCm.

## Local single-machine smoke test

For an end-to-end test on one host (SuperLink + ServerApp + two
SuperNode/ClientApp pairs) run [`../run-local.sh`](../run-local.sh). It
builds the three role Ryzers, brings everything up under `--network host`
on the loopback (with ClientAppIo ports offset by partition ID), submits
the quickstart-pytorch run, and tears down on exit.

The same three role Ryzers (`flower-superlink`, `flower-supernode`,
`flower-superexec`) are also what you run in a distributed deployment —
just set `SUPERLINK_IP` to the server's IP on the client boxes. See each
role Ryzer's README for the multi-machine flow.

## References

- [Flower documentation](https://flower.ai/docs/framework/)
- [Multi-machine Docker tutorial](https://flower.ai/docs/framework/docker/tutorial-deploy-on-multiple-machines.html)

Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
