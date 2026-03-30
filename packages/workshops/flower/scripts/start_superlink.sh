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

docker run --rm \
    --network host \
    --name "$SUPERLINK_NAME" \
    "$SUPERLINK_NAME:latest" \
    flower-superlink --insecure --isolation process
