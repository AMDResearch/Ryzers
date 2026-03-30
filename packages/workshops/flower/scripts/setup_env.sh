#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

source "$(dirname "$0")/env.sh"

echo "Setting up Flower demo environment..."

# Create workspace directory
mkdir -p "$FLOWER_WORKSPACE"

# Initialize Flower project using the SuperExec ryzer (which has flwr CLI)
echo "Initializing Flower project..."

if [ ! -d "$FLOWER_PROJECT" ]; then
    # Run flwr new command inside SuperExec ryzer with current user
    docker run --rm \
        -v "$FLOWER_WORKSPACE:/workspace" \
        -u "$(id -u):$(id -g)" \
        "$SUPEREXEC_SERVER_NAME:latest" \
        /bin/bash -c "cd /workspace && flwr new @flwrlabs/quickstart-pytorch"

    echo "✓ Flower project created at: $FLOWER_PROJECT"
else
    echo "✓ Flower project already exists: $FLOWER_PROJECT"
fi

# Create Flower config file in the project directory (so Docker can access it)
mkdir -p "$FLOWER_PROJECT/.flwr"
cat > "$FLOWER_PROJECT/.flwr/config.toml" << EOF
[superlink.local-deployment]
address = "127.0.0.1:9093"
insecure = true
EOF

# Fix permissions so Docker containers can access it
chmod 644 "$FLOWER_PROJECT/.flwr/config.toml"
chmod 755 "$FLOWER_PROJECT/.flwr"

# Also create in home directory for local use
mkdir -p ~/.flwr
cp "$FLOWER_PROJECT/.flwr/config.toml" ~/.flwr/config.toml
chmod 644 ~/.flwr/config.toml

echo "✓ Setup complete!"
echo ""
echo "Next step: ./start_demo.sh"
