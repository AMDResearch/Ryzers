#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLOWER_DIR="$(dirname "$SCRIPT_DIR")"

echo "Building all Flower containers..."
echo "=================================="

# Build superlink
echo ""
echo "Building superlink..."
cd "$FLOWER_DIR/superlink"
ryzers build --name superlink .

# Build supernode instances
echo ""
echo "Building supernode-1..."
cd "$FLOWER_DIR/supernode"
ryzers build --name supernode-1 .

echo ""
echo "Building supernode-2..."
ryzers build --name supernode-2 .

# Build superexec instances
echo ""
echo "Building superexec-serverapp..."
cd "$FLOWER_DIR/superexec"
ryzers build --name superexec-serverapp .

echo ""
echo "Building superexec-clientapp-1..."
ryzers build --name superexec-clientapp-1 .

echo ""
echo "Building superexec-clientapp-2..."
ryzers build --name superexec-clientapp-2 .

echo ""
echo "=================================="
echo "All Flower containers built successfully!"
echo ""
echo "You can now run the containers using:"
echo "  ./start_superlink.sh"
echo "  ./start_supernodes.sh"
echo "  ./start_superexecs.sh"
echo ""
echo "Or run all at once with:"
echo "  ./start_demos.sh"
