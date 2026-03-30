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

# Check if image was built
if ! docker image inspect flwr_superexec:0.0.1 &>/dev/null; then
    echo "Error: SuperExec image not found. Run ./build_containers.sh first."
    exit 1
fi

# Step 1: Launch SuperLink
echo "[1/6] Launching SuperLink (central coordinator)..."
docker run --rm \
  -p 9091:9091 -p 9092:9092 -p 9093:9093 \
  --network "$FLOWER_NETWORK" \
  --name superlink \
  --detach \
  flwr/superlink:$FLOWER_VERSION \
  --insecure \
  --isolation process

sleep 3

# Step 2: Launch SuperNode 1
echo "[2/6] Launching SuperNode 1..."
docker run --rm \
  -p 9094:9094 \
  --network "$FLOWER_NETWORK" \
  --name supernode-1 \
  --detach \
  flwr/supernode:$FLOWER_VERSION \
  --insecure \
  --superlink superlink:9092 \
  --node-config "partition-id=0 num-partitions=2" \
  --clientappio-api-address 0.0.0.0:9094 \
  --isolation process

# Step 3: Launch SuperNode 2
echo "[3/6] Launching SuperNode 2..."
docker run --rm \
  -p 9095:9095 \
  --network "$FLOWER_NETWORK" \
  --name supernode-2 \
  --detach \
  flwr/supernode:$FLOWER_VERSION \
  --insecure \
  --superlink superlink:9092 \
  --node-config "partition-id=1 num-partitions=2" \
  --clientappio-api-address 0.0.0.0:9095 \
  --isolation process

sleep 3

# Step 4: Launch ServerApp executor
echo "[4/6] Launching ServerApp executor..."
docker run --rm \
  --network "$FLOWER_NETWORK" \
  --name superexec-serverapp \
  --detach \
  flwr_superexec:0.0.1 \
  --insecure \
  --plugin-type serverapp \
  --appio-api-address superlink:9091

sleep 2

# Step 5: Launch ClientApp executors
echo "[5/6] Launching ClientApp executor 1..."
docker run --rm \
  --network "$FLOWER_NETWORK" \
  --name superexec-clientapp-1 \
  --detach \
  flwr_superexec:0.0.1 \
  --insecure \
  --plugin-type clientapp \
  --appio-api-address supernode-1:9094

echo "[5/6] Launching ClientApp executor 2..."
docker run --rm \
  --network "$FLOWER_NETWORK" \
  --name superexec-clientapp-2 \
  --detach \
  flwr_superexec:0.0.1 \
  --insecure \
  --plugin-type clientapp \
  --appio-api-address supernode-2:9095

sleep 2

# Step 6: Run Flower training
echo "[6/6] Starting Federated Learning training..."
echo ""
cd "$FLOWER_PROJECT"
flwr run . local-deployment --stream

echo ""
echo "========================================="
echo "  Demo complete!"
echo "========================================="
echo ""
echo "To cleanup: ./cleanup.sh"
