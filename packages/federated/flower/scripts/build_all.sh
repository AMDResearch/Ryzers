#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

echo "========================================"
echo "Building Flower Demo Components"
echo "========================================"
echo ""

# Step 1: Build ryzers_env base image (required for all ryzers)
echo "Step 1/7: Building ryzers_env base image..."
if docker images --format '{{.Repository}}' | grep -q '^ryzer_env$'; then
    echo "  ✓ ryzer_env already exists (skipping)"
else
    cd "$REPO_ROOT"
    ryzers build --base_path packages init/ryzer_env --name ryzer_env
    if [ $? -ne 0 ]; then
        echo "  ✗ Failed to build ryzer_env"
        exit 1
    fi
    echo "  ✓ ryzer_env built successfully"
fi
echo ""

# Step 2: Build superlink
echo "Step 2/7: Building superlink..."
cd "$REPO_ROOT"
ryzers build --base_path packages federated/flower superlink --name superlink
if [ $? -ne 0 ]; then
    echo "  ✗ Failed to build superlink"
    exit 1
fi
echo "  ✓ superlink built successfully"
echo ""

# Step 3: Build supernode-1
echo "Step 3/7: Building supernode-1..."
cd "$REPO_ROOT"
ryzers build --base_path packages federated/flower supernode-1 --name supernode-1
if [ $? -ne 0 ]; then
    echo "  ✗ Failed to build supernode-1"
    exit 1
fi
echo "  ✓ supernode-1 built successfully"
echo ""

# Step 4: Build supernode-2
echo "Step 4/7: Building supernode-2..."
cd "$REPO_ROOT"
ryzers build --base_path packages federated/flower supernode-2 --name supernode-2
if [ $? -ne 0 ]; then
    echo "  ✗ Failed to build supernode-2"
    exit 1
fi
echo "  ✓ supernode-2 built successfully"
echo ""

# Step 5: Build superexec-serverapp
echo "Step 5/7: Building superexec-serverapp..."
cd "$REPO_ROOT"
ryzers build --base_path packages federated/flower superexec --name superexec-serverapp
if [ $? -ne 0 ]; then
    echo "  ✗ Failed to build superexec-serverapp"
    exit 1
fi
echo "  ✓ superexec-serverapp built successfully"
echo ""

# Step 6: Build superexec-clientapp-1
echo "Step 6/7: Building superexec-clientapp-1..."
cd "$REPO_ROOT"
ryzers build --base_path packages federated/flower superexec --name superexec-clientapp-1
if [ $? -ne 0 ]; then
    echo "  ✗ Failed to build superexec-clientapp-1"
    exit 1
fi
echo "  ✓ superexec-clientapp-1 built successfully"
echo ""

# Step 7: Build superexec-clientapp-2
echo "Step 7/7: Building superexec-clientapp-2..."
cd "$REPO_ROOT"
ryzers build --base_path packages federated/flower superexec --name superexec-clientapp-2
if [ $? -ne 0 ]; then
    echo "  ✗ Failed to build superexec-clientapp-2"
    exit 1
fi
echo "  ✓ superexec-clientapp-2 built successfully"
echo ""

echo "========================================"
echo "✓ All Flower components built successfully!"
echo "========================================"
echo ""
echo "You can now run the demo with:"
echo "  $FLOWER_SCRIPTS/start_demos.sh"
echo ""
