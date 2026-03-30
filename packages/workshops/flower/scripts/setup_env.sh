#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

source "$(dirname "$0")/env.sh"

echo "Setting up Flower demo environment..."

# Create workspace directory with correct ownership
mkdir -p "$FLOWER_WORKSPACE"

# Fix ownership if it was created by Docker/root in a previous run
if [ "$(stat -c '%U' "$FLOWER_WORKSPACE" 2>/dev/null)" = "root" ]; then
    echo "Fixing workspace ownership (was owned by root)..."
    sudo chown -R "$(id -u):$(id -g)" "$FLOWER_WORKSPACE"
fi

# Ensure workspace has correct permissions
chmod 755 "$FLOWER_WORKSPACE"

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

# Fix directory permissions first to ensure we can write
chmod 755 "$FLOWER_PROJECT/.flwr" 2>/dev/null || true

# Remove old config if it exists and fix permissions
if [ -f "$FLOWER_PROJECT/.flwr/config.toml" ]; then
    chmod 644 "$FLOWER_PROJECT/.flwr/config.toml" 2>/dev/null || rm -f "$FLOWER_PROJECT/.flwr/config.toml"
fi

# Create new config file
cat > "$FLOWER_PROJECT/.flwr/config.toml" << EOF
[superlink.local-deployment]
address = "127.0.0.1:9093"
insecure = true
EOF

# Fix permissions so Docker containers can access it
chmod 644 "$FLOWER_PROJECT/.flwr/config.toml"

# Also create in home directory for local use
mkdir -p ~/.flwr

# Fix home directory permissions
if [ -f ~/.flwr/config.toml ]; then
    chmod 644 ~/.flwr/config.toml 2>/dev/null || rm -f ~/.flwr/config.toml
fi

cp "$FLOWER_PROJECT/.flwr/config.toml" ~/.flwr/config.toml
chmod 644 ~/.flwr/config.toml

echo "✓ Setup complete!"
echo ""
echo "Next step: ./start_demo.sh"
