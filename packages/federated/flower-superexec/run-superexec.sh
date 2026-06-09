#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Entry point for a flower-superexec process (the runner used for both
# ServerApp and ClientApp roles in Flower's process-isolation model).
#
# Invoke via:
#   ryzers run /ryzers/run-superexec.sh
#
# Required env vars:
#   FLOWER_PLUGIN_TYPE  — "serverapp" or "clientapp" (default serverapp)
#   FLOWER_APPIO_ADDR   — host:port of the paired SuperLink (ServerApp)
#                         or SuperNode (ClientApp).
#                         Defaults to 127.0.0.1:9091 (matches a local
#                         SuperLink). For ClientApp, set to your local
#                         SuperNode, e.g. 127.0.0.1:9094.

set -e

FLOWER_PLUGIN_TYPE="${FLOWER_PLUGIN_TYPE:-serverapp}"
FLOWER_INSECURE="${FLOWER_INSECURE:-1}"

case "${FLOWER_PLUGIN_TYPE}" in
  serverapp) DEFAULT_ADDR="127.0.0.1:9091" ;;
  clientapp) DEFAULT_ADDR="127.0.0.1:9094" ;;
  *)
    echo "FLOWER_PLUGIN_TYPE must be 'serverapp' or 'clientapp' (got: ${FLOWER_PLUGIN_TYPE})" >&2
    exit 2
    ;;
esac

FLOWER_APPIO_ADDR="${FLOWER_APPIO_ADDR:-${DEFAULT_ADDR}}"

ARGS=(
  --plugin-type "${FLOWER_PLUGIN_TYPE}"
  --appio-api-address "${FLOWER_APPIO_ADDR}"
)

if [ "${FLOWER_INSECURE}" = "1" ]; then
  ARGS+=(--insecure)
else
  ARGS+=(--root-certificates "${FLOWER_CA_CERT:-/app/certificates/ca.crt}")
fi

echo "Starting: flower-superexec ${ARGS[*]}"
exec flower-superexec "${ARGS[@]}"
