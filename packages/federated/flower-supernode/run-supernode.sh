#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Entry point for a SuperNode. Invoke via:
#   ryzers run /ryzers/run-supernode.sh
#
# Required:
#   SUPERLINK_IP        — IP of the remote SuperLink (defaults to 127.0.0.1)
#   FLOWER_PARTITION_ID — this node's partition (default 0)
#   FLOWER_NUM_PARTITIONS — total nodes in the federation (default 2)

set -e

SUPERLINK_IP="${SUPERLINK_IP:-127.0.0.1}"
FLOWER_INSECURE="${FLOWER_INSECURE:-1}"
FLOWER_PARTITION_ID="${FLOWER_PARTITION_ID:-0}"
FLOWER_NUM_PARTITIONS="${FLOWER_NUM_PARTITIONS:-2}"
FLOWER_ISOLATION="${FLOWER_ISOLATION:-process}"
FLOWER_CLIENTAPPIO="${FLOWER_CLIENTAPPIO:-0.0.0.0:9094}"

ARGS=(
  --superlink "${SUPERLINK_IP}:9092"
  --clientappio-api-address "${FLOWER_CLIENTAPPIO}"
  --isolation "${FLOWER_ISOLATION}"
  --node-config "partition-id=${FLOWER_PARTITION_ID} num-partitions=${FLOWER_NUM_PARTITIONS}"
)

if [ "${FLOWER_INSECURE}" = "1" ]; then
  ARGS+=(--insecure)
else
  ARGS+=(--root-certificates "${FLOWER_CA_CERT:-/app/certificates/ca.crt}")
fi

echo "Starting: flower-supernode ${ARGS[*]}"
exec flower-supernode "${ARGS[@]}"
