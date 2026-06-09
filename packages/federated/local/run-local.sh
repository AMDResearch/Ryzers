#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# One-shot local smoke test for the Flower federation: builds the three
# role images (if needed), brings the compose stack up, runs the
# quickstart-pytorch example end-to-end, then tears the stack down.
#
# Usage: bash run-local.sh

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
APP_DIR="${HERE}/../flower-superexec/app"

cd "${REPO}"

echo "==> Building role images (chained on flower-base)"
ryzers build flower-base flower-superlink
ryzers build flower-base flower-supernode
ryzers build flower-base flower-superexec

cd "${HERE}"
mkdir -p state

echo "==> Bringing the federation up"
docker compose up -d

cleanup() {
  echo "==> Tearing down"
  docker compose down -v || true
}
trap cleanup EXIT

echo "==> Waiting for SuperLink ExecApi (127.0.0.1:9091)"
for i in $(seq 1 30); do
  (echo >/dev/tcp/127.0.0.1/9091) >/dev/null 2>&1 && break
  sleep 1
done

echo "==> Submitting quickstart-pytorch run via flwr CLI"
# The host must have `flwr` installed: pip install "flwr==1.26.1"
# Federation name "local" must be configured in app/pyproject.toml — see
# packages/federated/local/README.md for the snippet.
flwr run "${APP_DIR}" local

echo "==> Smoke test complete"
