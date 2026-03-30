#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

source "$(dirname "$0")/env.sh"

# Activate virtualenv if it exists
if [ -f "$FLOWER_VENV/bin/activate" ]; then
    source "$FLOWER_VENV/bin/activate"
fi

echo "========================================="
echo "  Flower SuperNode 1 (Client Manager)"
echo "========================================="

# Change to Ryzers root (ryzers run must run from repo root)
RYZERS_ROOT="$(cd "$FLOWER_PATH/../../.." && pwd)"
cd "$RYZERS_ROOT"

# Modify the run script to add container name and port
# ryzers run generates script but we need unique container name per SuperNode
docker run --rm \
    --network "$FLOWER_NETWORK" \
    --name "$SUPERNODE1_NAME" \
    -p 9094:9094 \
    "$SUPERNODE1_NAME:latest" \
    flower-supernode \
    --insecure \
    --superlink "$SUPERLINK_NAME:9092" \
    --node-config "partition-id=0 num-partitions=2" \
    --clientappio-api-address 0.0.0.0:9094 \
    --isolation process
