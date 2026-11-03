#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Entry point for the SuperLink. Invoke via:
#   ryzers run /ryzers/run-superlink.sh
#
# All knobs are env vars (defaults make insecure local testing work
# out of the box). Override by exporting them in your shell before
# calling `ryzers run` — config.yaml propagates them via shell expansion.

set -e

FLOWER_INSECURE="${FLOWER_INSECURE:-1}"
FLOWER_ISOLATION="${FLOWER_ISOLATION:-process}"
FLOWER_STATE_DB="${FLOWER_STATE_DB:-/app/state/state.db}"

ARGS=(--isolation "${FLOWER_ISOLATION}" --database "${FLOWER_STATE_DB}")

if [ "${FLOWER_INSECURE}" = "1" ]; then
  ARGS+=(--insecure)
else
  ARGS+=(
    --ssl-ca-certfile "${FLOWER_CA_CERT:-/app/certificates/ca.crt}"
    --ssl-certfile    "${FLOWER_SERVER_CERT:-/app/certificates/server.pem}"
    --ssl-keyfile     "${FLOWER_SERVER_KEY:-/app/certificates/server.key}"
  )
fi

mkdir -p "$(dirname "${FLOWER_STATE_DB}")"

echo "Starting: flower-superlink ${ARGS[*]}"
exec flower-superlink "${ARGS[@]}"
