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

# Change to Ryzers root directory (ryzers run must run from repo root)
RYZERS_ROOT="$(cd "$FLOWER_PATH/../../.." && pwd)"
pushd "$RYZERS_ROOT"

# Step 1: Launch SuperLink
echo "[1/6] Launching SuperLink (central coordinator)..."
ryzers run --name "$SUPERLINK_NAME" \
    --network host \
    --detach \
    -- flower-superlink --insecure --isolation process

sleep 3

# Step 2: Launch SuperNode 1
echo "[2/6] Launching SuperNode 1..."
ryzers run --name "$SUPERNODE1_NAME" \
    --network host \
    --detach \
    -- flower-supernode \
    --insecure \
    --superlink 127.0.0.1:9092 \
    --node-config "partition-id=0 num-partitions=2" \
    --clientappio-api-address 0.0.0.0:9094 \
    --isolation process

# Step 3: Launch SuperNode 2
echo "[3/6] Launching SuperNode 2..."
ryzers run --name "$SUPERNODE2_NAME" \
    --network host \
    --detach \
    -- flower-supernode \
    --insecure \
    --superlink 127.0.0.1:9092 \
    --node-config "partition-id=1 num-partitions=2" \
    --clientappio-api-address 0.0.0.0:9095 \
    --isolation process

sleep 3

# Step 4: Launch ServerApp executor
echo "[4/6] Launching ServerApp executor..."
ryzers run --name "$SUPEREXEC_SERVER_NAME" \
    --network host \
    -v "$FLOWER_PROJECT:/app" \
    --detach \
    -- flower-superexec \
    --insecure \
    --executor-config "app-dir=/app" \
    --executor flwr.superexec.deployment:executor

sleep 2

# Step 5: Launch ClientApp executors
echo "[5/6] Launching ClientApp executor 1..."
ryzers run --name "$SUPEREXEC_CLIENT1_NAME" \
    --network host \
    -v "$FLOWER_PROJECT:/app" \
    --detach \
    -- flower-superexec \
    --insecure \
    --executor-config "app-dir=/app node-id=0" \
    --executor flwr.superexec.deployment:executor

echo "[5/6] Launching ClientApp executor 2..."
ryzers run --name "$SUPEREXEC_CLIENT2_NAME" \
    --network host \
    -v "$FLOWER_PROJECT:/app" \
    --detach \
    -- flower-superexec \
    --insecure \
    --executor-config "app-dir=/app node-id=1" \
    --executor flwr.superexec.deployment:executor

sleep 2

# Step 6: Run Flower training (using SuperExec ryzer with flwr CLI)
echo "[6/6] Starting Federated Learning training..."
echo ""

ryzers run --name "$SUPEREXEC_SERVER_NAME" \
    --network host \
    -v "$FLOWER_PROJECT:/app" \
    -- /bin/bash -c "cd /app && flwr run . local-deployment --stream"

popd

echo ""
echo "========================================="
echo "  Demo complete!"
echo "========================================="
echo ""
echo "To cleanup: ./cleanup.sh"
