#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

source "$(dirname "$0")/env.sh"

echo "Cleaning up Flower containers..."

# Stop all SuperExec containers
docker stop $(docker ps -q --filter ancestor=flwr_superexec:0.0.1) 2>/dev/null || true

# Stop SuperNodes and SuperLink
docker stop supernode-1 supernode-2 superlink 2>/dev/null || true

echo "✓ All containers stopped"
echo ""
echo "To remove the Docker network (optional):"
echo "  docker network rm $FLOWER_NETWORK"
