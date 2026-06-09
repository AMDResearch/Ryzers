# Flower SuperLink Docker Setup

The SuperLink is the central coordinator in a Flower deployment. It accepts
connections from SuperNodes (Fleet API, port 9092), runs the control plane
(Exec API, port 9091), and persists run state to `/app/state`.

This Ryzer runs on the **server machine** and is the first component you
start when bringing up a federation.

## Build

```sh
ryzers build flower-base flower-superlink
```

## Run (insecure — testing only)

The default `CMD` runs `test_flower-superlink.sh`, which starts the
SuperLink with `--insecure` and verifies it binds port 9092. To run it
properly, override the CMD:

```sh
ryzers run flower-superlink --insecure
```

## Run (with TLS — for real deployments)

1. Generate certs on the server machine, specifying its routable IP:

   ```sh
   SUPERLINK_IP=192.168.2.33 \
     bash packages/federated/flower-superlink/gen-certs.sh
   ```

   This produces `ca.crt`, `server.pem`, `server.key` under
   `./workspace/flower/superlink-certificates/`, which the Ryzer's
   `config.yaml` mounts read-only at `/app/certificates/`.

2. Copy `ca.crt` to every client machine (the SuperNodes will need it).

3. Start the SuperLink:

   ```sh
   ryzers run flower-superlink \
     --ssl-ca-certfile=/app/certificates/ca.crt \
     --ssl-certfile=/app/certificates/server.pem \
     --ssl-keyfile=/app/certificates/server.key \
     --database=/app/state/state.db \
     --isolation=process
   ```

## Ports

| Port | API | Used by |
|------|-----|---------|
| 9091 | ExecApi | local `flower-superexec --plugin-type serverapp` |
| 9092 | FleetApi | remote SuperNodes |
| 9093 | ServerAppIo | (legacy / TLS control) |

## References

- [SuperLink reference](https://flower.ai/docs/framework/ref-api-cli.html#flower-superlink)
- [Multi-machine Docker tutorial](https://flower.ai/docs/framework/docker/tutorial-deploy-on-multiple-machines.html)

Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
