#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

source "$(dirname "$0")/env.sh"

# Activate virtualenv if it exists
if [ -f "$FLOWER_VENV/bin/activate" ]; then
    source "$FLOWER_VENV/bin/activate"
fi

echo "========================================="
echo "  Flower SuperExec (ServerApp)"
echo "========================================="

docker run --rm \
    --network host \
    --name "$SUPEREXEC_SERVER_NAME" \
    -v "$FLOWER_PROJECT:/app" \
    "$SUPEREXEC_SERVER_NAME:latest" \
    flower-superexec \
    --insecure \
    --executor-config "app-dir=/app" \
    --executor flwr.superexec.deployment:executor
