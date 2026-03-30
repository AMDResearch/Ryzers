#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

source "$(dirname "$0")/env.sh"

echo "Cleaning up Flower containers..."

# Change to Ryzers root directory (ryzers stop must run from repo root)
RYZERS_ROOT="$(cd "$FLOWER_PATH/../../.." && pwd)"
pushd "$RYZERS_ROOT"

# Stop all Flower ryzer containers
ryzers stop "$SUPERLINK_NAME" 2>/dev/null || true
ryzers stop "$SUPERNODE1_NAME" 2>/dev/null || true
ryzers stop "$SUPERNODE2_NAME" 2>/dev/null || true
ryzers stop "$SUPEREXEC_SERVER_NAME" 2>/dev/null || true
ryzers stop "$SUPEREXEC_CLIENT1_NAME" 2>/dev/null || true
ryzers stop "$SUPEREXEC_CLIENT2_NAME" 2>/dev/null || true

popd

echo "✓ All containers stopped"
