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

# Note: Flower project creation moved to build_containers.sh
# (needs SuperExec container to be built first)

echo "✓ Setup complete!"
echo ""
echo "Next step: ./build_containers.sh"
