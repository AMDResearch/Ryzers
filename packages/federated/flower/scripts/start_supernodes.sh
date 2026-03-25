#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

source $(dirname "$0")/env.sh

echo "Starting Flower Supernodes..."

# Note: For multi-instance deployments, build separate images for each instance:
#   ryzers build supernode --name supernode-1
#   ryzers build supernode --name supernode-2

# Start supernode-1
# The base configuration comes from supernode/config.yaml
ryzers run --name supernode-1 "--insecure --superlink superlink:9092 --node-config partition-id=0 num-partitions=2 --clientappio-api-address 0.0.0.0:9094 --isolation process" &

echo "Started supernode-1"
sleep 2

# Start supernode-2
ryzers run --name supernode-2 "--insecure --superlink superlink:9092 --node-config partition-id=0 num-partitions=2 --clientappio-api-address 0.0.0.0:9095 --isolation process" &

echo "Started supernode-2"
sleep 2
echo "Both supernodes are running"

exec bash
