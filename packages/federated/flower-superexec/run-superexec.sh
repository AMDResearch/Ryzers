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
#   FLOWER_PLUGIN_TYPE   — "serverapp", "clientapp", or "submit"
#                          (default serverapp)
#   FLOWER_PARTITION_ID  — for clientapp, offsets the default
#                          SuperNode ClientAppIo port (9094 + ID).
#   FLOWER_APPIO_ADDR    — host:port of the paired SuperLink (ServerApp)
#                          or SuperNode (ClientApp). Default depends on
#                          plugin type.
#
# "submit" plugin type is a one-shot helper: it runs `flwr run /app local`
# against the local SuperLink (using the `local` federation pre-baked
# into /app/pyproject.toml) and then exits.

set -e

FLOWER_PLUGIN_TYPE="${FLOWER_PLUGIN_TYPE:-serverapp}"
FLOWER_INSECURE="${FLOWER_INSECURE:-1}"
FLOWER_PARTITION_ID="${FLOWER_PARTITION_ID:-0}"

if [ "${FLOWER_PLUGIN_TYPE}" = "submit" ]; then
  # `--stream` keeps this process attached to the run and returns only once
  # the run has finished (i.e. the ServerApp has written final_model.pt to
  # disk). This lets the local orchestrator (run-local.sh) detect completion
  # and tear the federation down afterwards instead of leaving it running.
  echo "Submitting: flwr run /app local --stream"
  exec flwr run /app local --stream
fi

case "${FLOWER_PLUGIN_TYPE}" in
  serverapp) DEFAULT_ADDR="127.0.0.1:9091" ;;
  clientapp) DEFAULT_ADDR="127.0.0.1:$((9094 + FLOWER_PARTITION_ID))" ;;
  *)
    echo "FLOWER_PLUGIN_TYPE must be 'serverapp', 'clientapp', or 'submit' (got: ${FLOWER_PLUGIN_TYPE})" >&2
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
