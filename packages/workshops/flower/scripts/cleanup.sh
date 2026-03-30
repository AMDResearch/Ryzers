#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

source "$(dirname "$0")/env.sh"

# Activate virtualenv if it exists
if [ -f "$FLOWER_VENV/bin/activate" ]; then
    source "$FLOWER_VENV/bin/activate"
fi

echo "Cleaning up Flower containers..."

# Stop all Flower containers
docker stop "$SUPERLINK_NAME" 2>/dev/null || true
docker stop "$SUPERNODE1_NAME" 2>/dev/null || true
docker stop "$SUPERNODE2_NAME" 2>/dev/null || true
docker stop "$SUPEREXEC_SERVER_NAME" 2>/dev/null || true
docker stop "$SUPEREXEC_CLIENT1_NAME" 2>/dev/null || true
docker stop "$SUPEREXEC_CLIENT2_NAME" 2>/dev/null || true

echo "✓ All containers stopped"
echo ""
echo "To remove the Docker network:"
echo "  docker network rm $FLOWER_NETWORK"
