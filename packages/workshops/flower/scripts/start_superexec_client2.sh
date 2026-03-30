#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

source "$(dirname "$0")/env.sh"

# Activate virtualenv if it exists
if [ -f "$FLOWER_VENV/bin/activate" ]; then
    source "$FLOWER_VENV/bin/activate"
fi

echo "========================================="
echo "  Flower SuperExec (ClientApp 2)"
echo "========================================="

docker run --rm \
    --network host \
    --name "$SUPEREXEC_CLIENT2_NAME" \
    -v "$FLOWER_PROJECT:/app" \
    "$SUPEREXEC_CLIENT2_NAME:latest" \
    flower-superexec \
    --insecure \
    --executor-config "app-dir=/app node-id=1" \
    --executor flwr.superexec.deployment:executor
