#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

source $(dirname "$0")/env.sh

echo "Starting Flower Supernodes..."

# Check if base supernode image exists
if ! docker images --format '{{.Repository}}' | grep -q '^supernode$'; then
    echo "ERROR: Base supernode image not found!"
    echo "Please build it first with: ryzers build flower supernode --name supernode"
    exit 1
fi

# Get the run flags from the base supernode image
# We'll use the generated run script as a template
RUNSCRIPT="$REPO_ROOT/ryzers.run.supernode.sh"
if [ ! -f "$RUNSCRIPT" ]; then
    echo "ERROR: Run script not found at $RUNSCRIPT"
    echo "Please build the base image first: ryzers build flower supernode --name supernode"
    exit 1
fi

# Extract the docker run flags from the generated script
# This captures all the flags like --network, --device, etc.
BASE_FLAGS=$(grep "^docker run" "$RUNSCRIPT" | sed 's/docker run //' | sed 's/ supernode.*//')

# Check and restart supernode-1 if running
if docker ps --format '{{.Names}}' | grep -q '^supernode-1$'; then
    echo "Supernode-1 is already running. Restarting..."
    docker stop supernode-1
    docker rm supernode-1 2>/dev/null || true
fi

# Start supernode-1 using docker run directly
echo "Starting supernode-1..."
docker run --name supernode-1 \
    -p 9094:9094 \
    --detach \
    $BASE_FLAGS \
    supernode \
    --insecure \
    --superlink superlink:9092 \
    --node-config "partition-id=0 num-partitions=2" \
    --clientappio-api-address 0.0.0.0:9094 \
    --isolation process

echo "Started supernode-1"
sleep 2

# Check and restart supernode-2 if running
if docker ps --format '{{.Names}}' | grep -q '^supernode-2$'; then
    echo "Supernode-2 is already running. Restarting..."
    docker stop supernode-2
    docker rm supernode-2 2>/dev/null || true
fi

# Start supernode-2 using docker run directly
echo "Starting supernode-2..."
docker run --name supernode-2 \
    -p 9095:9095 \
    --detach \
    $BASE_FLAGS \
    supernode \
    --insecure \
    --superlink superlink:9092 \
    --node-config "partition-id=1 num-partitions=2" \
    --clientappio-api-address 0.0.0.0:9095 \
    --isolation process

echo "Started supernode-2"
sleep 2
echo "Both supernodes are running"

exec bash
