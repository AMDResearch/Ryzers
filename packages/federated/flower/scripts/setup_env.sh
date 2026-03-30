#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

source "$(dirname "$0")/env.sh"

echo "Setting up Flower demo environment..."

# Create workspace directory
mkdir -p "$FLOWER_WORKSPACE"

# Create Docker network for Flower components
if ! docker network inspect "$FLOWER_NETWORK" &>/dev/null; then
    echo "Creating Docker network: $FLOWER_NETWORK"
    docker network create --driver bridge "$FLOWER_NETWORK"
else
    echo "Docker network already exists: $FLOWER_NETWORK"
fi

# Initialize Flower project
cd "$FLOWER_WORKSPACE"
if [ ! -d "$FLOWER_PROJECT" ]; then
    echo "Initializing Flower project..."
    flwr new @flwrlabs/quickstart-pytorch
    echo "Flower project created at: $FLOWER_PROJECT"
else
    echo "Flower project already exists: $FLOWER_PROJECT"
fi

# Create Flower config file
mkdir -p ~/.flwr
cat > ~/.flwr/config.toml << EOF
[superlink.local-deployment]
address = "127.0.0.1:9093"
insecure = true
EOF

echo "✓ Setup complete!"
echo ""
echo "Next step: ./build_containers.sh"
