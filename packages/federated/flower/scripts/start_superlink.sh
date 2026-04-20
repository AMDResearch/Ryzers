#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

source $(dirname "$0")/env.sh

echo "Starting Flower Superlink..."

# Check if superlink is already running
if docker ps --format '{{.Names}}' | grep -q '^superlink$'; then
    echo "Superlink container is already running. Restarting..."
    docker stop superlink
    docker rm superlink 2>/dev/null || true
    echo "Stopped existing superlink container"
fi

# Start the superlink container using ryzers run
# Network configuration is set in superlink/config.yaml (docker_network: flwr-network)
ryzers run --name superlink "--insecure --isolation process"

echo "Superlink started successfully"

exec bash
