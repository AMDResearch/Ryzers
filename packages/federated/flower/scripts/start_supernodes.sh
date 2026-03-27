#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

source $(dirname "$0")/env.sh

echo "Starting Flower Supernodes..."

# Note: Build separate images for each instance:
#   ryzers build supernode-1 --name supernode-1
#   ryzers build supernode-2 --name supernode-2

# Check and restart supernode-1 if running
if docker ps --format '{{.Names}}' | grep -q '^supernode-1$'; then
    echo "Supernode-1 is already running. Restarting..."
    docker stop supernode-1
    docker rm supernode-1 2>/dev/null || true
fi

# Start supernode-1 (port mapping 9094:9094 configured in config.yaml)
ryzers run --name supernode-1 "--insecure --superlink superlink:9092 --node-config partition-id=0 num-partitions=2 --clientappio-api-address 0.0.0.0:9094 --isolation process" &

echo "Started supernode-1"
sleep 2

# Check and restart supernode-2 if running
if docker ps --format '{{.Names}}' | grep -q '^supernode-2$'; then
    echo "Supernode-2 is already running. Restarting..."
    docker stop supernode-2
    docker rm supernode-2 2>/dev/null || true
fi

# Start supernode-2 (port mapping 9095:9095 configured in config.yaml)
ryzers run --name supernode-2 "--insecure --superlink superlink:9092 --node-config partition-id=1 num-partitions=2 --clientappio-api-address 0.0.0.0:9095 --isolation process" &

echo "Started supernode-2"
sleep 2
echo "Both supernodes are running"

exec bash
