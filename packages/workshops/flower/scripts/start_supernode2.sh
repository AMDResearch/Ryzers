#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

source "$(dirname "$0")/env.sh"

# Activate virtualenv if it exists
if [ -f "$FLOWER_VENV/bin/activate" ]; then
    source "$FLOWER_VENV/bin/activate"
fi

echo "========================================="
echo "  Flower SuperNode 2 (Client Manager)"
echo "========================================="

docker run --rm \
    --network host \
    --name "$SUPERNODE2_NAME" \
    "$SUPERNODE2_NAME:latest" \
    flower-supernode \
    --insecure \
    --superlink 127.0.0.1:9092 \
    --node-config "partition-id=1 num-partitions=2" \
    --clientappio-api-address 0.0.0.0:9095 \
    --isolation process
