#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

source "$(dirname "$0")/env.sh"

echo "========================================="
echo "  Flower SuperNode 1 (Client Manager)"
echo "========================================="

docker run --rm \
    --network host \
    --name "$SUPERNODE1_NAME" \
    "$SUPERNODE1_NAME:latest" \
    flower-supernode \
    --insecure \
    --superlink 127.0.0.1:9092 \
    --node-config "partition-id=0 num-partitions=2" \
    --clientappio-api-address 0.0.0.0:9094 \
    --isolation process
