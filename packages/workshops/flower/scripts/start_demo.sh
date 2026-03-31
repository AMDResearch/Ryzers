#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

source "$(dirname "$0")/env.sh"

# Activate virtualenv if it exists
if [ -f "$FLOWER_VENV/bin/activate" ]; then
    source "$FLOWER_VENV/bin/activate"
fi

echo "========================================="
echo "  Flower Federated Learning Demo"
echo "========================================="
echo ""

# Check if setup was run
if [ ! -d "$FLOWER_PROJECT" ]; then
    echo "Error: Flower project not found. Run ./setup_env.sh first."
    exit 1
fi

# Export FLOWER_PROJECT for use in docker run flags (config.yaml uses $FLOWER_PROJECT)
export FLOWER_PROJECT

# Change to Ryzers root directory
RYZERS_ROOT="$(cd "$FLOWER_PATH/../../.." && pwd)"
cd "$RYZERS_ROOT"

# Cleanup function for graceful shutdown
cleanup() {
    echo ""
    echo "Stopping all Flower containers..."
    docker stop "$SUPERLINK_NAME" 2>/dev/null || true
    docker stop "$SUPERNODE1_NAME" 2>/dev/null || true
    docker stop "$SUPERNODE2_NAME" 2>/dev/null || true
    docker stop "$SUPEREXEC_SERVER_NAME" 2>/dev/null || true
    docker stop "$SUPEREXEC_CLIENT1_NAME" 2>/dev/null || true
    docker stop "$SUPEREXEC_CLIENT2_NAME" 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# Step 1: Launch SuperLink in background
echo "[1/6] Launching SuperLink (background)..."
bash "ryzers.run.${SUPERLINK_NAME}.sh" "flower-superlink --insecure --isolation process" &
SUPERLINK_PID=$!

sleep 3

# Step 2: Launch SuperNode 1 in background
echo "[2/6] Launching SuperNode 1 (background)..."
bash "ryzers.run.${SUPERNODE1_NAME}.sh" "flower-supernode --insecure --superlink $SUPERLINK_NAME:9092 --node-config 'partition-id=0 num-partitions=2' --clientappio-api-address 0.0.0.0:9094 --isolation process" &
SUPERNODE1_PID=$!

# Step 3: Launch SuperNode 2 in background
echo "[3/6] Launching SuperNode 2 (background)..."
bash "ryzers.run.${SUPERNODE2_NAME}.sh" "flower-supernode --insecure --superlink $SUPERLINK_NAME:9092 --node-config 'partition-id=1 num-partitions=2' --clientappio-api-address 0.0.0.0:9095 --isolation process" &
SUPERNODE2_PID=$!

sleep 3

# Step 4: Launch ServerApp executor in background
echo "[4/6] Launching ServerApp executor (background)..."
bash "ryzers.run.${SUPEREXEC_SERVER_NAME}.sh" "flower-superexec --insecure --plugin-type serverapp --appio-api-address $SUPERLINK_NAME:9091" &
SUPEREXEC_SERVER_PID=$!

sleep 2

# Step 5: Launch ClientApp executors in background
echo "[5/6] Launching ClientApp executor 1 (background)..."
bash "ryzers.run.${SUPEREXEC_CLIENT1_NAME}.sh" "flower-superexec --insecure --plugin-type clientapp --appio-api-address $SUPERNODE1_NAME:9094" &
SUPEREXEC_CLIENT1_PID=$!

echo "[5/6] Launching ClientApp executor 2 (background)..."
bash "ryzers.run.${SUPEREXEC_CLIENT2_NAME}.sh" "flower-superexec --insecure --plugin-type clientapp --appio-api-address $SUPERNODE2_NAME:9095" &
SUPEREXEC_CLIENT2_PID=$!

sleep 2

# Show running containers
echo ""
echo "All components started. Running containers:"
docker ps --filter "name=flower-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ""

# Step 6: Run Flower training in this terminal (foreground/interactive)
echo "[6/6] Starting Federated Learning training..."
echo ""

# For the training command, we need interactive mode (-it), but build script sets -d
# Create a temporary interactive version of the run script
RUNSCRIPT="ryzers.run.${SUPEREXEC_SERVER_NAME}.sh"
TEMP_RUNSCRIPT="${RUNSCRIPT}.interactive.tmp"
sed 's/ -d / -it /g' "$RUNSCRIPT" > "$TEMP_RUNSCRIPT"
chmod +x "$TEMP_RUNSCRIPT"

# Run the interactive training command
# Install project dependencies and run flwr (pass as single quoted argument)
bash "$TEMP_RUNSCRIPT" 'sh -c "cd /app && pip install -q -e . && flwr run . local-deployment --stream"'

# Cleanup temporary script
rm -f "$TEMP_RUNSCRIPT"

echo ""
echo "========================================="
echo "  Demo complete!"
echo "========================================="
echo ""
echo "To cleanup: ./cleanup.sh"

# Cleanup on exit
cleanup
