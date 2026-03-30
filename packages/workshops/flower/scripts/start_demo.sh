#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

source "$(dirname "$0")/env.sh"

echo "========================================="
echo "  Flower Federated Learning Demo"
echo "========================================="
echo ""

# Check if setup was run
if [ ! -d "$FLOWER_PROJECT" ]; then
    echo "Error: Flower project not found. Run ./setup_env.sh first."
    exit 1
fi

export FLOWER_SCRIPTS=$FLOWER_SCRIPTS

# Step 1: Launch SuperLink in its own terminal
echo "[1/6] Launching SuperLink in new terminal..."
gnome-terminal --tab --title="Flower SuperLink" -- bash -c 'echo -ne "\033]0;Flower SuperLink\007"; '"$FLOWER_SCRIPTS"'/start_superlink.sh; exec bash'

sleep 3

# Step 2: Launch SuperNode 1 in its own terminal
echo "[2/6] Launching SuperNode 1 in new terminal..."
gnome-terminal --tab --title="Flower SuperNode 1" -- bash -c 'echo -ne "\033]0;Flower SuperNode 1\007"; '"$FLOWER_SCRIPTS"'/start_supernode1.sh; exec bash'

# Step 3: Launch SuperNode 2 in its own terminal
echo "[3/6] Launching SuperNode 2 in new terminal..."
gnome-terminal --tab --title="Flower SuperNode 2" -- bash -c 'echo -ne "\033]0;Flower SuperNode 2\007"; '"$FLOWER_SCRIPTS"'/start_supernode2.sh; exec bash'

sleep 3

# Step 4: Launch ServerApp executor in its own terminal
echo "[4/6] Launching ServerApp executor in new terminal..."
gnome-terminal --tab --title="Flower ServerApp" -- bash -c 'echo -ne "\033]0;Flower ServerApp\007"; '"$FLOWER_SCRIPTS"'/start_superexec_server.sh; exec bash'

sleep 2

# Step 5: Launch ClientApp executors in their own terminals
echo "[5/6] Launching ClientApp executor 1 in new terminal..."
gnome-terminal --tab --title="Flower ClientApp 1" -- bash -c 'echo -ne "\033]0;Flower ClientApp 1\007"; '"$FLOWER_SCRIPTS"'/start_superexec_client1.sh; exec bash'

echo "[5/6] Launching ClientApp executor 2 in new terminal..."
gnome-terminal --tab --title="Flower ClientApp 2" -- bash -c 'echo -ne "\033]0;Flower ClientApp 2\007"; '"$FLOWER_SCRIPTS"'/start_superexec_client2.sh; exec bash'

sleep 2

# Step 6: Run Flower training in this terminal (interactive)
echo "[6/6] Starting Federated Learning training..."
echo ""

# Set FLWR_HOME to the project's .flwr directory so it finds the config
docker run --rm -it \
    --network "$FLOWER_NETWORK" \
    -v "$FLOWER_PROJECT:/app" \
    -e FLWR_HOME=/app/.flwr \
    "$SUPEREXEC_SERVER_NAME:latest" \
    /bin/bash -c "cd /app && flwr run . local-deployment --stream"

echo ""
echo "========================================="
echo "  Demo complete!"
echo "========================================="
echo ""
echo "To cleanup: ./cleanup.sh"
