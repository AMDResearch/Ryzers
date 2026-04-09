#!/bin/bash

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

echo "========================================"
echo "  Flower SuperLink"
echo "========================================"
echo ""

# Check Flower installation
echo "Checking Flower installation..."
python3 -c "import flwr; print(f'Flower version: {flwr.__version__}')"
echo ""

# Verify flower-superlink command exists
echo "Checking flower-superlink command..."
which flower-superlink
echo ""

echo "SuperLink ready. Run with:"
echo "  ryzers run flower-superlink --insecure --isolation process"
echo ""
echo "Ports:"
echo "  9091: ServerAppIO API"
echo "  9092: Fleet API"
echo "  9093: Control API"
