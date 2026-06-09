# Flower SuperLink Docker Setup

The SuperLink is the central coordinator in a Flower deployment. It
accepts connections from the ServerApp superexec (ServerAppIo, port
9091), SuperNodes (Fleet API, port 9092), and the `flwr run` submitter
(Exec API, port 9093). Run state is persisted to `/app/state`.

This Ryzer runs on the **server machine** and is the first component you
start when bringing up a federation. The container uses `--network host`
so all three ports bind directly on the host — local components reach
them via `127.0.0.1`, remote ones via the server's IP.

For a single-machine smoke test that brings up SuperLink + ServerApp +
two SuperNode/ClientApp pairs and submits the example run, use
[`../run-local.sh`](../run-local.sh).

## Build

```sh
ryzers build flower-base flower-superlink
```

## Run

SuperLink flags are driven by environment variables declared in
`config.yaml` (with shell-expansion defaults). Export them in your
shell before `ryzers run` to override.

### Insecure (local testing)

```sh
ryzers run
```

### With TLS (real deployment)

1. Generate certs on the server machine, specifying its routable IP:

   ```sh
   SUPERLINK_IP=192.168.2.33 \
     bash packages/federated/flower-superlink/gen-certs.sh
   ```

   This produces `ca.crt`, `server.pem`, `server.key` under
   `./workspace/flower/superlink-certificates/`, which the Ryzer's
   `config.yaml` mounts read-only at `/app/certificates/`.

2. Copy `ca.crt` to every client machine (the SuperNodes will need it).

3. Launch the SuperLink with TLS:

   ```sh
   export FLOWER_INSECURE=0
   ryzers run
   ```

   To override the cert paths inside the container, also export
   `FLOWER_CA_CERT`, `FLOWER_SERVER_CERT`, `FLOWER_SERVER_KEY`.

### Env-var reference

| Variable | Default | Purpose |
|----------|---------|---------|
| `FLOWER_INSECURE` | `1` | `1` = `--insecure`; `0` = enable TLS flags |
| `FLOWER_ISOLATION` | `process` | Passed to `--isolation` |
| `FLOWER_STATE_DB` | `/app/state/state.db` | Persisted run state |

## Ports

| Port | API | Used by |
|------|-----|---------|
| 9091 | ServerAppIo | local `flower-superexec --plugin-type serverapp` |
| 9092 | Fleet | remote SuperNodes |
| 9093 | Exec | `flwr run` submitter (and the `submit` plugin-type) |

## References

- [SuperLink reference](https://flower.ai/docs/framework/ref-api-cli.html#flower-superlink)
- [Multi-machine Docker tutorial](https://flower.ai/docs/framework/docker/tutorial-deploy-on-multiple-machines.html)

Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
