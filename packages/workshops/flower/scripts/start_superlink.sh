#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

source "$(dirname "$0")/env.sh"

echo "========================================="
echo "  Flower SuperLink (Coordinator)"
echo "========================================="

docker run --rm \
    --network host \
    --name "$SUPERLINK_NAME" \
    "$SUPERLINK_NAME:latest" \
    flower-superlink --insecure --isolation process
