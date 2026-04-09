#!/bin/bash

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

echo "========================================"
echo "  Flower SuperNode"
echo "========================================"
echo ""

# Check Flower installation
echo "Checking Flower installation..."
python3 -c "import flwr; print(f'Flower version: {flwr.__version__}')"
echo ""

# Verify flower-supernode command exists
echo "Checking flower-supernode command..."
which flower-supernode
echo ""

echo "SuperNode ready. Example usage:"
echo ""
echo "  # SuperNode 1 (partition 0 of 2)"
echo "  ryzers run flower-supernode --insecure \\"
echo "    --superlink <superlink-host>:9092 \\"
echo "    --node-config 'partition-id=0 num-partitions=2' \\"
echo "    --clientappio-api-address 0.0.0.0:9094 \\"
echo "    --isolation process"
echo ""
echo "  # SuperNode 2 (partition 1 of 2)"
echo "  ryzers run flower-supernode --insecure \\"
echo "    --superlink <superlink-host>:9092 \\"
echo "    --node-config 'partition-id=1 num-partitions=2' \\"
echo "    --clientappio-api-address 0.0.0.0:9095 \\"
echo "    --isolation process"
