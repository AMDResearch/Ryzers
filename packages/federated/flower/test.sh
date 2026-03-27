#!/bin/bash

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

echo "Testing Flower installation..."
python -c "import flwr;print('flwr version:', flwr.__version__)"

echo ""
echo "Checking quickstart project..."
if [ -d "/ryzers/quickstart-pytorch" ]; then
    echo "✓ quickstart-pytorch project found"
    ls -la /ryzers/quickstart-pytorch
else
    echo "✗ quickstart-pytorch project not found"
    exit 1
fi

echo ""
echo "Checking Flower configuration..."
flwr config list
