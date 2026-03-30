#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

source "$(dirname "$0")/env.sh"

# Activate virtualenv if it exists
if [ -f "$FLOWER_VENV/bin/activate" ]; then
    source "$FLOWER_VENV/bin/activate"
fi

echo "========================================="
echo "  Flower SuperLink (Coordinator)"
echo "========================================="

# Change to Ryzers root (ryzers run must run from repo root)
RYZERS_ROOT="$(cd "$FLOWER_PATH/../../.." && pwd)"
cd "$RYZERS_ROOT"

# Use docker run with bridge network (ryzers run doesn't support --network flag)
docker run --rm \
    --network "$FLOWER_NETWORK" \
    --name "$SUPERLINK_NAME" \
    "$SUPERLINK_NAME:latest" \
    flower-superlink --insecure --isolation process
