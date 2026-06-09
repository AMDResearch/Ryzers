# Flower Local Single-Machine Deployment

Run the entire federation — SuperLink, ServerApp, and two SuperNode +
ClientApp pairs — on a single host for smoke-testing and development.
Mirrors the upstream multi-machine tutorial structure but collapses
everything onto one box on a shared docker bridge network.

## Topology

```
                   ┌─────────────┐
                   │  superlink  │
                   └──┬───┬───┬──┘
            9091 ─────┘   │   │
            (ExecApi)     │   │ 9092 (FleetApi)
                          │   │
                    ┌─────┘   └─────┐
                    │               │
              ┌─────────┐       ┌─────────┐
              │serverapp│       │supernode│ ×2
              └─────────┘       └────┬────┘
                                     │ 9094
                                     │
                                ┌────────┐
                                │clientapp│ ×2
                                └────────┘
```

## Prerequisites

Build the three role images (they all layer on `flower-base`):

```sh
ryzers build flower-base flower-superlink
ryzers build flower-base flower-supernode
ryzers build flower-base flower-superexec
```

The compose file references them by name (`flower-superlink:latest`
etc.) so the local docker daemon must have all three tagged.

## Bring up the federation

```sh
cd packages/federated/local
docker compose up        # add -d to detach
```

Tear down with `docker compose down -v` (the `-v` clears the SuperLink
state volume too).

## Submit a training run

From the host (requires `pip install "flwr==1.26.1"` on the host):

1. Add a federation entry pointing at the local SuperLink in
   `../flower-superexec/app/pyproject.toml`:

   ```toml
   [tool.flwr.federations.local]
   address = "127.0.0.1:9093"
   insecure = true
   ```

2. Submit the run:

   ```sh
   flwr run ../flower-superexec/app local
   ```

You should see two clients pick up partitions, train one round each,
report metrics back to the ServerApp, and exit after three rounds (per
the example's `num-server-rounds = 3` default).

## One-shot smoke test

`run-local.sh` builds the images (if needed), brings the stack up, runs
the example, then tears down:

```sh
bash run-local.sh
```

## Notes

- All components run with `FLOWER_INSECURE=1` (no TLS). For a TLS
  rehearsal, generate certs with `flower-superlink/gen-certs.sh` and
  add `FLOWER_INSECURE=0` plus the cert volume mounts to the compose
  file.
- Each `clientapp-*` service mounts `/dev/kfd` and `/dev/dri` so PyTorch
  can use the ROCm GPU. If your host has no AMD GPU, comment those
  blocks out — the example will fall back to CPU.
- `partition-id` is set per SuperNode (0 and 1). Add more
  `supernode-N` + `clientapp-N` pairs and bump `num-partitions`
  accordingly to scale the federation.

Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
