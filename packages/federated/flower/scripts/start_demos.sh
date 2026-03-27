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

# Navigate to the quickstart project directory and run deployment
echo "Launching local-deployment in a new terminal..."
gnome-terminal --tab --title="Flower Local Deployment" -- bash -c '
echo -ne "\033]0;Flower Local Deployment\007"
cd "'"$FLOWER_QUICKSTART_DIR"'" || {
    echo "ERROR: Failed to change to directory '"$FLOWER_QUICKSTART_DIR"'"
    read -p "Press Enter to close..."
    exit 1
}

echo "========================================"
echo "Flower Local Deployment"
echo "========================================"
echo ""
echo "Current directory: $(pwd)"
echo "Running: flwr run . local-deployment --stream"
echo ""

# Run the deployment and show all output
set -x
flwr run . local-deployment --stream
EXIT_CODE=$?
set +x

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Deployment completed successfully!"
else
    echo "✗ Deployment failed with exit code: $EXIT_CODE"
fi
echo "========================================"
echo ""
echo "Review the output above to see training progress and results."
echo ""
read -p "Press Enter to close this window..."
'

echo ""
echo "Local deployment started in new terminal tab"
echo "Check the 'Flower Local Deployment' tab to view progress and training output"
