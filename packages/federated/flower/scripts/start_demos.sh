#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/env.sh"

export FLOWER_SCRIPTS="$SCRIPT_DIR"

# Create Docker network if it doesn't exist
if ! docker network ls | grep -q flwr-network; then
    echo "Creating Docker bridge network: flwr-network"
    docker network create --driver bridge flwr-network
else
    echo "Docker network flwr-network already exists"
fi
echo ""

# Launch each component in a separate terminal tab
gnome-terminal --tab --title="Superlink" -- bash -c 'echo -ne "\033]0;Superlink\007"; '"$FLOWER_SCRIPTS"'/start_superlink.sh; exec bash'

sleep 5
gnome-terminal --tab --title="Supernodes" -- bash -c 'echo -ne "\033]0;Supernodes\007"; '"$FLOWER_SCRIPTS"'/start_supernodes.sh; exec bash'

sleep 5
gnome-terminal --tab --title="Superexecs" -- bash -c 'echo -ne "\033]0;Superexecs\007"; '"$FLOWER_SCRIPTS"'/start_superexecs.sh; exec bash'

echo "All Flower components started in separate terminal tabs"
echo "To view logs, check each terminal tab"
echo ""

# Wait for all components to be ready
echo "Waiting for all components to be ready..."
sleep 10

# Verify required Docker images exist
echo "Verifying Flower component images..."
REQUIRED_IMAGES=("superlink" "supernode-1" "supernode-2" "superexec-serverapp" "superexec-clientapp-1" "superexec-clientapp-2")
MISSING_IMAGES=()

for image in "${REQUIRED_IMAGES[@]}"; do
    if docker images --format '{{.Repository}}' | grep -q "^${image}$"; then
        echo "  ✓ $image image exists"
    else
        echo "  ✗ $image image missing"
        MISSING_IMAGES+=("$image")
    fi
done

if [ ${#MISSING_IMAGES[@]} -gt 0 ]; then
    echo ""
    echo "ERROR: Missing required Docker images!"
    echo "Please build all required images by running:"
    echo ""
    echo "  $FLOWER_SCRIPTS/build_all.sh"
    echo ""
    exit 1
fi

echo "All required images exist!"
echo ""

# Verify all containers are running
echo "Verifying all Flower components are running..."
REQUIRED_CONTAINERS=("superlink" "supernode-1" "supernode-2" "superexec-serverapp" "superexec-clientapp-1" "superexec-clientapp-2")
ALL_RUNNING=true

for container in "${REQUIRED_CONTAINERS[@]}"; do
    if docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
        echo "  ✓ $container is running"
    else
        echo "  ✗ $container is NOT running"
        ALL_RUNNING=false
    fi
done

if [ "$ALL_RUNNING" = false ]; then
    echo ""
    echo "ERROR: Not all required containers are running!"
    echo "Please check the terminal tabs for error messages."
    exit 1
fi

echo ""
echo "All containers verified!"
echo ""

# Execute the local-deployment on the host
echo "========================================"
echo "Running Flower local-deployment..."
echo "========================================"
echo ""

# Configure flwr for local-deployment (always ensure it exists)
mkdir -p ~/.flwr
if ! grep -q "\[superlink.local-deployment\]" ~/.flwr/config.toml 2>/dev/null; then
    echo "Configuring Flower local-deployment federation..."
    cat >> ~/.flwr/config.toml <<EOF
[defaults]
federation = "local-deployment"

[superlink.local-deployment]
address = "127.0.0.1:9093"
insecure = true

EOF
fi

# Check if quickstart project exists on host, if not create it
if [ ! -d "$FLOWER_QUICKSTART_DIR" ]; then
    echo "Creating quickstart-pytorch project on host..."
    cd "$FLOWER_PATH"
    flwr new @flwrlabs/quickstart-pytorch
    cd quickstart-pytorch
    # Remove torch dependencies as they're in the containers
    sed -i '/torch=/d;/torchvision=/d' pyproject.toml

    # Remove legacy federation configuration if present
    sed -i '/\[tool.flwr.federations\]/,/^$/d' pyproject.toml

    pip install -e .
fi

# Verify quickstart directory exists
if [ ! -d "$FLOWER_QUICKSTART_DIR" ]; then
    echo "ERROR: Quickstart directory not found at $FLOWER_QUICKSTART_DIR"
    echo "This should have been created during setup. Please check for errors above."
    exit 1
fi

# Verify superlink is accessible from host
echo "Verifying superlink connection from host..."
timeout 10 bash -c 'until nc -z localhost 9093; do echo "Waiting for superlink..."; sleep 1; done' || {
    echo "ERROR: Cannot connect to superlink at localhost:9093"
    echo "Check that the superlink container is running and port 9093 is exposed"
    exit 1
}
echo "✓ Superlink is accessible from host"
echo ""

# Navigate to quickstart directory
cd "$FLOWER_QUICKSTART_DIR" || {
    echo "ERROR: Failed to change to directory $FLOWER_QUICKSTART_DIR"
    exit 1
}

echo "Current directory: $(pwd)"
echo "Running: flwr run . local-deployment --stream"
echo ""
echo "Starting federated learning training..."
echo "This will show real-time output from the ServerApp."
echo ""

# Run the deployment directly in this terminal
flwr run . local-deployment --stream
EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Deployment completed successfully!"
else
    echo "✗ Deployment failed with exit code: $EXIT_CODE"
fi
echo "========================================"
