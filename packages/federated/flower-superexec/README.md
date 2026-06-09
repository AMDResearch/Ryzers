# Flower SuperExec Docker Setup

`flower-superexec` is Flower's process runner. The same binary runs both
the **ServerApp** (server-side aggregation logic) and the **ClientApp**
(client-side training), selected with `--plugin-type {serverapp,clientapp}`.

This Ryzer is the only one in the federation that actually executes
PyTorch training, so it sits on top of the ROCm/PyTorch base image and
inherits GPU access. It ships with the `quickstart-pytorch` example app
preinstalled at `/app/`.

## Build

```sh
ryzers build flower-base flower-superexec
```

## Run — ServerApp (one per federation, on the server machine)

```sh
ryzers run flower-superexec \
  --insecure \
  --plugin-type serverapp \
  --appio-api-address ${SUPERLINK_HOST:-127.0.0.1}:9091
```

## Run — ClientApp (one per SuperNode, on each client machine)

```sh
ryzers run flower-superexec \
  --insecure \
  --plugin-type clientapp \
  --appio-api-address ${SUPERNODE_HOST:-127.0.0.1}:9094
```

## Putting it together (mirrors the upstream tutorial)

**On the server machine** (one SuperLink + one ServerApp superexec):

```sh
ryzers build flower-base flower-superlink
ryzers build flower-base flower-superexec   # separate image, ServerApp role

# Terminal 1 — SuperLink
ryzers run flower-superlink --insecure --isolation process

# Terminal 2 — ServerApp superexec
ryzers run flower-superexec \
  --insecure --plugin-type serverapp \
  --appio-api-address 127.0.0.1:9091
```

**On each client machine** (one SuperNode + one ClientApp superexec):

```sh
ryzers build flower-base flower-supernode
ryzers build flower-base flower-superexec

# Terminal 1 — SuperNode (connects to remote SuperLink)
ryzers run flower-supernode \
  --insecure \
  --superlink ${SUPERLINK_IP}:9092 \
  --clientappio-api-address 0.0.0.0:9094 \
  --node-config "partition-id=0 num-partitions=2" \
  --isolation process

# Terminal 2 — ClientApp superexec (runs the PyTorch training)
ryzers run flower-superexec \
  --insecure --plugin-type clientapp \
  --appio-api-address 127.0.0.1:9094
```

Then from any machine with the `flwr` CLI installed:

```sh
flwr run /app remote-deployment      # see config.toml in upstream tutorial
```

For TLS (production), see `flower-superlink/gen-certs.sh` and pass
`--root-certificates /app/certificates/ca.crt` to the superexec instead
of `--insecure`.

## Bundled example

`/app/` contains a copy of [`examples/quickstart-pytorch`](https://github.com/adap/flower/tree/v1.26.1/examples/quickstart-pytorch),
modified to drop the `torch==2.8.0` pin so the ROCm PyTorch from the
base image is used unchanged.

## References

- [SuperExec / process isolation](https://flower.ai/docs/framework/how-to-deploy-flower-server-using-process-isolation.html)
- [Multi-machine Docker tutorial](https://flower.ai/docs/framework/docker/tutorial-deploy-on-multiple-machines.html)

Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
