#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

source "$(dirname "$0")/env.sh"

echo "Building SuperExec Docker image..."

cd "$FLOWER_PROJECT"

# Copy template Dockerfile
cp "$FLOWER_PATH/templates/superexec.Dockerfile" .

# Build SuperExec image
docker build -f superexec.Dockerfile -t flwr_superexec:0.0.1 .

echo "✓ SuperExec image built successfully: flwr_superexec:0.0.1"
echo ""
echo "Next step: ./start_demo.sh"
