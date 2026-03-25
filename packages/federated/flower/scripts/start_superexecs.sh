#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

source $(dirname "$0")/env.sh

echo "Starting Flower Superexecs..."

# Note: For multi-instance deployments, build separate images for each instance:
#   ryzers build superexec --name superexec-serverapp
#   ryzers build superexec --name superexec-clientapp-1
#   ryzers build superexec --name superexec-clientapp-2

# Start serverapp superexec
ryzers run --name superexec-serverapp "--insecure --plugin-type serverapp --appio-api-address superlink:9091" &

echo "Started superexec-serverapp"
sleep 2

# Start clientapp-1 superexec
ryzers run --name superexec-clientapp-1 "--insecure --plugin-type clientapp --appio-api-address supernode-1:9094" &

echo "Started superexec-clientapp-1"
sleep 2

# Start clientapp-2 superexec
ryzers run --name superexec-clientapp-2 "--insecure --plugin-type clientapp --appio-api-address supernode-2:9095" &

echo "Started superexec-clientapp-2"
sleep 2
echo "All superexecs are running"

exec bash
