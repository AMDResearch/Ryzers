#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

source "$(dirname "$0")/env.sh"

echo "Building Flower ryzer containers..."

# Change to Ryzers root directory (ryzers build must run from repo root)
RYZERS_ROOT="$(cd "$FLOWER_PATH/../../.." && pwd)"
pushd "$RYZERS_ROOT"

# Build ryzer_env first (base environment)
echo "[0/3] Building ryzer_env (base environment)..."
ryzers build ryzer_env --name ryzer_env

# Build SuperLink ryzer
echo "[1/3] Building SuperLink..."
ryzers build flower-superlink --name "$SUPERLINK_NAME"

# Build SuperNode ryzer
echo "[2/3] Building SuperNode..."
ryzers build flower-supernode --name "$SUPERNODE1_NAME"
ryzers build flower-supernode --name "$SUPERNODE2_NAME"

# Build SuperExec ryzer
echo "[3/3] Building SuperExec..."
ryzers build flower-superexec --name "$SUPEREXEC_SERVER_NAME"
ryzers build flower-superexec --name "$SUPEREXEC_CLIENT1_NAME"
ryzers build flower-superexec --name "$SUPEREXEC_CLIENT2_NAME"

popd

echo ""
echo "✓ All Flower ryzers built successfully!"
echo ""
echo "Next step: ./setup_env.sh"
