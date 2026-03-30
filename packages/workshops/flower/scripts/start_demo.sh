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

# Step 1: Launch SuperLink
echo "[1/6] Launching SuperLink (central coordinator)..."
docker run --rm -d \
    --network host \
    --name "$SUPERLINK_NAME" \
    "$SUPERLINK_NAME:latest" \
    flower-superlink --insecure --isolation process

sleep 3

# Step 2: Launch SuperNode 1
echo "[2/6] Launching SuperNode 1..."
docker run --rm -d \
    --network host \
    --name "$SUPERNODE1_NAME" \
    "$SUPERNODE1_NAME:latest" \
    flower-supernode \
    --insecure \
    --superlink 127.0.0.1:9092 \
    --node-config "partition-id=0 num-partitions=2" \
    --clientappio-api-address 0.0.0.0:9094 \
    --isolation process

# Step 3: Launch SuperNode 2
echo "[3/6] Launching SuperNode 2..."
docker run --rm -d \
    --network host \
    --name "$SUPERNODE2_NAME" \
    "$SUPERNODE2_NAME:latest" \
    flower-supernode \
    --insecure \
    --superlink 127.0.0.1:9092 \
    --node-config "partition-id=1 num-partitions=2" \
    --clientappio-api-address 0.0.0.0:9095 \
    --isolation process

sleep 3

# Step 4: Launch ServerApp executor
echo "[4/6] Launching ServerApp executor..."
docker run --rm -d \
    --network host \
    --name "$SUPEREXEC_SERVER_NAME" \
    -v "$FLOWER_PROJECT:/app" \
    "$SUPEREXEC_SERVER_NAME:latest" \
    flower-superexec \
    --insecure \
    --executor-config "app-dir=/app" \
    --executor flwr.superexec.deployment:executor

sleep 2

# Step 5: Launch ClientApp executors
echo "[5/6] Launching ClientApp executor 1..."
docker run --rm -d \
    --network host \
    --name "$SUPEREXEC_CLIENT1_NAME" \
    -v "$FLOWER_PROJECT:/app" \
    "$SUPEREXEC_CLIENT1_NAME:latest" \
    flower-superexec \
    --insecure \
    --executor-config "app-dir=/app node-id=0" \
    --executor flwr.superexec.deployment:executor

echo "[5/6] Launching ClientApp executor 2..."
docker run --rm -d \
    --network host \
    --name "$SUPEREXEC_CLIENT2_NAME" \
    -v "$FLOWER_PROJECT:/app" \
    "$SUPEREXEC_CLIENT2_NAME:latest" \
    flower-superexec \
    --insecure \
    --executor-config "app-dir=/app node-id=1" \
    --executor flwr.superexec.deployment:executor

sleep 2

# Step 6: Run Flower training (using SuperExec ryzer with flwr CLI)
echo "[6/6] Starting Federated Learning training..."
echo ""

docker run --rm -it \
    --network host \
    -v "$FLOWER_PROJECT:/app" \
    "$SUPEREXEC_SERVER_NAME:latest" \
    /bin/bash -c "cd /app && flwr run . local-deployment --stream"

echo ""
echo "========================================="
echo "  Demo complete!"
echo "========================================="
echo ""
echo "To cleanup: ./cleanup.sh"
