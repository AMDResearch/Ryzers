#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLOWER_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$FLOWER_DIR/../../.." && pwd)"

echo "Building all Flower containers..."
echo "=================================="

# Build base image if it doesn't exist
if ! docker image inspect ryzer_env &> /dev/null; then
    echo ""
    echo "Base image 'ryzer_env' not found. Building it first..."
    cd "$REPO_ROOT"
    ryzers build ryzer_env --name ryzer_env
fi

# Build flower base image
echo ""
echo "Building flower base image..."
cd "$REPO_ROOT"
ryzers build flower --name flower

# Build superlink
echo ""
echo "Building superlink..."
cd "$REPO_ROOT"
ryzers build flower superlink --name superlink

# Build supernode instances
echo ""
echo "Building supernode-1..."
cd "$REPO_ROOT"
ryzers build flower supernode --name supernode-1

echo ""
echo "Building supernode-2..."
ryzers build flower supernode --name supernode-2

# Build superexec instances
echo ""
echo "Building superexec-serverapp..."
cd "$REPO_ROOT"
ryzers build flower superexec --name superexec-serverapp

echo ""
echo "Building superexec-clientapp-1..."
ryzers build flower superexec --name superexec-clientapp-1

echo ""
echo "Building superexec-clientapp-2..."
ryzers build flower superexec --name superexec-clientapp-2

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
