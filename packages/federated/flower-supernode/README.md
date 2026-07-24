# Flower SuperNode Docker Setup

The SuperNode runs on each **client machine** in a Flower deployment. It
connects to the SuperLink's Fleet API (port 9092) and exposes a local
ClientAppIo socket (port 9094) for the paired `flower-superexec
--plugin-type clientapp` to attach to.

Pair this Ryzer with `flower-superexec` on every client host. The
SuperNode does the federation plumbing; the superexec runs the actual
PyTorch/ROCm training.

The container uses `--network host`, so `SUPERLINK_IP` is the only
network knob you usually need to set. The ClientAppIo socket defaults
to `0.0.0.0:$((9094 + FLOWER_PARTITION_ID))` so multiple SuperNodes can
share a host for local testing without colliding (see
[`../run-local.sh`](../run-local.sh)).

## Build

```sh
ryzers build --name flower-supernode flower-base flower-supernode
```

## Run

SuperNode flags are driven by environment variables declared in
`config.yaml` (with shell-expansion defaults). Export them in your
shell before `ryzers run` to override.

### Insecure (testing)

```sh
export SUPERLINK_IP=192.168.2.33
export FLOWER_PARTITION_ID=0
export FLOWER_NUM_PARTITIONS=2
ryzers run
```

### With TLS

Place `ca.crt` at `./workspace/flower/superlink-certificates/ca.crt`
(the volume mount in `config.yaml` exposes it at `/app/certificates/ca.crt`),
then:

```sh
export SUPERLINK_IP=192.168.2.33
export FLOWER_INSECURE=0
export FLOWER_PARTITION_ID=0
export FLOWER_NUM_PARTITIONS=2
ryzers run
```

### Env-var reference

| Variable | Default | Purpose |
|----------|---------|---------|
| `SUPERLINK_IP` | `127.0.0.1` | Routable IP of the SuperLink host |
| `FLOWER_INSECURE` | `1` | `1` = `--insecure`; `0` = TLS via `FLOWER_CA_CERT` |
| `FLOWER_PARTITION_ID` | `0` | Unique partition for this node (0..N-1) |
| `FLOWER_NUM_PARTITIONS` | `2` | Total clients across the federation |
| `FLOWER_CLIENTAPPIO` | `0.0.0.0:$((9094 + FLOWER_PARTITION_ID))` | Local socket the paired ClientApp connects to (auto-offset for co-located nodes) |
| `FLOWER_ISOLATION` | `process` | Passed to `--isolation` |

## References

- [SuperNode reference](https://flower.ai/docs/framework/ref-api-cli.html#flower-supernode)

Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
