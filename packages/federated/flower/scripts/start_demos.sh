#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/env.sh"

export FLOWER_SCRIPTS="$SCRIPT_DIR"

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

# Check if quickstart project exists on host, if not create it
if [ ! -d "$FLOWER_QUICKSTART_DIR" ]; then
    echo "Creating quickstart-pytorch project on host..."
    cd "$FLOWER_PATH"
    flwr new @flwrlabs/quickstart-pytorch
    cd quickstart-pytorch
    # Remove torch dependencies as they're in the containers
    sed -i '/torch=/d;/torchvision=/d' pyproject.toml
    pip install -e .

    # Configure flwr for local-deployment
    mkdir -p ~/.flwr
    cat > ~/.flwr/config.toml <<EOF
[superlink.local-deployment]
address = "127.0.0.1:9093"
insecure = true
EOF
fi

# Navigate to the quickstart project directory and run deployment
cd "$FLOWER_QUICKSTART_DIR"
flwr run . local-deployment --stream
