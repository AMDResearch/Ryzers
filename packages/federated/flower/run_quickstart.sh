#!/bin/bash

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

echo "========================================"
echo "Flower Quickstart - PyTorch Example"
echo "========================================"
echo ""

# Navigate to the quickstart project directory
cd /ryzers/quickstart-pytorch

# Display configuration
echo "Flower configuration:"
flwr config list
echo ""

# Check if superlink is accessible
echo "Checking SuperLink connection at 127.0.0.1:9093..."
timeout 5 bash -c 'until nc -z 127.0.0.1 9093; do sleep 1; done' 2>/dev/null && echo "SuperLink is accessible!" || echo "Warning: SuperLink not detected. Make sure it's running."
echo ""

# Run the quickstart project
echo "Running Flower federated learning quickstart..."
echo "Command: flwr run . local-deployment --stream"
echo ""

flwr run . local-deployment --stream
