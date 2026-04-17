#!/bin/bash

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set -e

echo "Testing O3DE installation..."

# Verify o3de is installed
if ! command -v o3de &> /dev/null; then
    echo "FAIL: o3de command not found"
    exit 1
fi

echo "o3de found at: $(which o3de)"

# Check o3de version/help (non-blocking)
echo "Checking O3DE CLI..."
o3de --help > /dev/null 2>&1 || true

# Verify critical O3DE components exist
echo "Verifying O3DE installation paths..."
O3DE_PATH="/opt/O3DE"
if [[ -d "$O3DE_PATH" ]]; then
    echo "O3DE installed at: $O3DE_PATH"
else
    echo "Warning: O3DE path not found at $O3DE_PATH"
fi

# Check for Vulkan support
echo "Checking Vulkan support..."
if command -v vulkaninfo &> /dev/null; then
    vulkaninfo --summary 2>/dev/null | head -20 || echo "Vulkan info not available (may need GPU)"
else
    echo "vulkaninfo not found"
fi

echo "SUCCESS: O3DE installation test passed"
echo "Note: For interactive use, run: o3de"
exit 0
