#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

source $(dirname "$0")/env.sh

echo "========================================"
echo "Stopping Flower Demo Components"
echo "========================================"
echo ""

# Stop all Flower containers
echo "Stopping Flower containers..."

# Stop superexec containers
docker stop superexec-serverapp 2>/dev/null && echo "Stopped superexec-serverapp" || echo "superexec-serverapp not running"
docker stop superexec-clientapp-1 2>/dev/null && echo "Stopped superexec-clientapp-1" || echo "superexec-clientapp-1 not running"
docker stop superexec-clientapp-2 2>/dev/null && echo "Stopped superexec-clientapp-2" || echo "superexec-clientapp-2 not running"

# Stop supernode containers
docker stop supernode-1 2>/dev/null && echo "Stopped supernode-1" || echo "supernode-1 not running"
docker stop supernode-2 2>/dev/null && echo "Stopped supernode-2" || echo "supernode-2 not running"

# Stop superlink container
docker stop superlink 2>/dev/null && echo "Stopped superlink" || echo "superlink not running"

echo ""
echo "All Flower containers stopped"
echo ""

# Always try to remove the network (it will only succeed if no containers are using it)
echo "Removing Docker network 'flwr-network'..."
docker network rm flwr-network 2>/dev/null && echo "✓ Removed flwr-network" || echo "Note: Network may still be in use or not found"
echo ""

# Ask if user wants to clean up quickstart project
read -p "Remove quickstart-pytorch project directory? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -d "$FLOWER_QUICKSTART_DIR" ]; then
        rm -rf "$FLOWER_QUICKSTART_DIR"
        echo "Removed $FLOWER_QUICKSTART_DIR"
    else
        echo "Quickstart directory not found"
    fi
fi

# Ask if user wants to clean up config
read -p "Remove Flower configuration (~/.flwr/config.toml)? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -f ~/.flwr/config.toml ]; then
        rm ~/.flwr/config.toml
        echo "Removed ~/.flwr/config.toml"
    else
        echo "Config file not found"
    fi
fi

echo ""
echo "========================================"
echo "Cleanup complete"
echo "========================================"
