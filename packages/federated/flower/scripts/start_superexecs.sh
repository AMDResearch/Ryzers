#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

source $(dirname "$0")/env.sh

echo "Starting Flower Superexecs..."

# Check if base superexec image exists
if ! docker images --format '{{.Repository}}' | grep -q '^superexec$'; then
    echo "ERROR: Base superexec image not found!"
    echo "Please build it first with: ryzers build flower superexec --name superexec"
    exit 1
fi

# Get the run flags from the base superexec image
RUNSCRIPT="$REPO_ROOT/ryzers.run.superexec.sh"
if [ ! -f "$RUNSCRIPT" ]; then
    echo "ERROR: Run script not found at $RUNSCRIPT"
    echo "Please build the base image first: ryzers build flower superexec --name superexec"
    exit 1
fi

# Extract the docker run flags from the generated script
BASE_FLAGS=$(grep "^docker run" "$RUNSCRIPT" | sed 's/docker run //' | sed 's/ superexec.*//')

# Check and restart superexec-serverapp if running
if docker ps --format '{{.Names}}' | grep -q '^superexec-serverapp$'; then
    echo "Superexec-serverapp is already running. Restarting..."
    docker stop superexec-serverapp
    docker rm superexec-serverapp 2>/dev/null || true
fi

# Start serverapp superexec using docker run directly
echo "Starting superexec-serverapp..."
docker run --name superexec-serverapp \
    --detach \
    $BASE_FLAGS \
    superexec \
    --insecure \
    --plugin-type serverapp \
    --appio-api-address superlink:9091

echo "Started superexec-serverapp"
sleep 2

# Check and restart superexec-clientapp-1 if running
if docker ps --format '{{.Names}}' | grep -q '^superexec-clientapp-1$'; then
    echo "Superexec-clientapp-1 is already running. Restarting..."
    docker stop superexec-clientapp-1
    docker rm superexec-clientapp-1 2>/dev/null || true
fi

# Start clientapp-1 superexec using docker run directly
echo "Starting superexec-clientapp-1..."
docker run --name superexec-clientapp-1 \
    --detach \
    $BASE_FLAGS \
    superexec \
    --insecure \
    --plugin-type clientapp \
    --appio-api-address supernode-1:9094

echo "Started superexec-clientapp-1"
sleep 2

# Check and restart superexec-clientapp-2 if running
if docker ps --format '{{.Names}}' | grep -q '^superexec-clientapp-2$'; then
    echo "Superexec-clientapp-2 is already running. Restarting..."
    docker stop superexec-clientapp-2
    docker rm superexec-clientapp-2 2>/dev/null || true
fi

# Start clientapp-2 superexec using docker run directly
echo "Starting superexec-clientapp-2..."
docker run --name superexec-clientapp-2 \
    --detach \
    $BASE_FLAGS \
    superexec \
    --insecure \
    --plugin-type clientapp \
    --appio-api-address supernode-2:9095

echo "Started superexec-clientapp-2"
sleep 2
echo "All superexecs are running"

exec bash
