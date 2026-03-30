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

# Change to Ryzers root (ryzers run must run from repo root)
RYZERS_ROOT="$(cd "$FLOWER_PATH/../../.." && pwd)"
cd "$RYZERS_ROOT"

# Use docker run with bridge network and dynamic volume mount
docker run --rm \
    --network "$FLOWER_NETWORK" \
    --name "$SUPEREXEC_CLIENT2_NAME" \
    -v "$FLOWER_PROJECT:/app" \
    --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --group-add video --group-add render \
    "$SUPEREXEC_CLIENT2_NAME:latest" \
    flower-superexec \
    --insecure \
    --plugin-type clientapp \
    --appio-api-address "$SUPERNODE2_NAME:9095"
