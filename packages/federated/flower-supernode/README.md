# Flower SuperNode Docker Setup

The SuperNode runs on each **client machine** in a Flower deployment. It
connects to the SuperLink's Fleet API (port 9092) and exposes a local
ClientAppIo socket (port 9094) for the paired `flower-superexec
--plugin-type clientapp` to attach to.

Pair this Ryzer with `flower-superexec` on every client host. The
SuperNode does the federation plumbing; the superexec runs the actual
PyTorch/ROCm training.

## Build

```sh
ryzers build flower-base flower-supernode
```

## Run

The default `CMD` runs `test_flower-supernode.sh`, which only validates
the CLI. To start a real SuperNode, override the CMD and point it at
your SuperLink:

### Insecure (testing only)

```sh
ryzers run flower-supernode \
  --superlink ${SUPERLINK_IP}:9092 \
  --clientappio-api-address 0.0.0.0:9094 \
  --insecure \
  --node-config "partition-id=0 num-partitions=2" \
  --isolation process
```

### With TLS (using `ca.crt` copied from the server machine)

Place `ca.crt` at `./workspace/flower/superlink-certificates/ca.crt`
(the volume mount in `config.yaml` makes it visible inside the container
at `/app/certificates/ca.crt`), then:

```sh
ryzers run flower-supernode \
  --superlink ${SUPERLINK_IP}:9092 \
  --clientappio-api-address 0.0.0.0:9094 \
  --root-certificates /app/certificates/ca.crt \
  --node-config "partition-id=0 num-partitions=2" \
  --isolation process
```

`partition-id` should be unique per client; `num-partitions` is the total
number of clients across the federation.

## References

- [SuperNode reference](https://flower.ai/docs/framework/ref-api-cli.html#flower-supernode)

Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
