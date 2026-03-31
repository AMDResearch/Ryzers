#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

source "$(dirname "$0")/env.sh"

echo "Setting up Flower demo environment..."

# Create virtualenv if it doesn't exist
if [ ! -d "$FLOWER_VENV" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "$FLOWER_VENV"
    echo "✓ Virtual environment created at: $FLOWER_VENV"
else
    echo "✓ Virtual environment already exists: $FLOWER_VENV"
fi

# Activate virtualenv for this script
if [ -f "$FLOWER_VENV/bin/activate" ]; then
    source "$FLOWER_VENV/bin/activate"
fi

# Install Ryzers into the virtualenv
echo "Installing Ryzers into virtualenv..."
RYZERS_ROOT="$(cd "$FLOWER_PATH/../../.." && pwd)"
pip install -q -e "$RYZERS_ROOT"
echo "✓ Ryzers installed"

# Create Docker bridge network for Flower components
if ! docker network inspect "$FLOWER_NETWORK" &>/dev/null; then
    echo "Creating Docker network: $FLOWER_NETWORK"
    docker network create --driver bridge "$FLOWER_NETWORK"
else
    echo "✓ Docker network already exists: $FLOWER_NETWORK"
fi

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
    # Using docker run directly as volume path is dynamic
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

# Create new config file (using container name on bridge network)
# Note: Using non-quoted EOF to expand variables
cat > "$FLOWER_PROJECT/.flwr/config.toml" <<EOF
[superlink.local-deployment]
address = "$SUPERLINK_NAME:9093"
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

# Install project dependencies locally (for verification)
echo "Installing project dependencies..."
cd "$FLOWER_PROJECT"
pip install -q -e .
cd - > /dev/null

echo "✓ Setup complete!"
echo ""
echo "Next step: ./build_containers.sh"
