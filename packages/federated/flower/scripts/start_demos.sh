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

# Execute the local-deployment
echo "========================================"
echo "Running Flower local-deployment..."
echo "========================================"
echo ""

# Check if we're in a container or on the host
if [ -d "/ryzers/quickstart-pytorch" ]; then
    # Running inside the flower container
    cd /ryzers/quickstart-pytorch
    flwr run . local-deployment --stream
else
    # Running from host - need to execute in the flower container
    echo "Launching local-deployment in flower container..."
    docker run --rm -it \
        --network host \
        flower \
        bash -c "cd /ryzers/quickstart-pytorch && flwr run . local-deployment --stream"
fi
