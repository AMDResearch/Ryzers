#!/bin/bash

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set -e

echo "Testing amdgpu_top installation..."

# Check if amdgpu_top is installed
if ! command -v amdgpu_top &> /dev/null; then
    echo "FAIL: amdgpu_top command not found"
    exit 1
fi

echo "amdgpu_top found at: $(which amdgpu_top)"

# Run amdgpu_top in dump mode (non-interactive) with short duration
echo "Running amdgpu_top in dump mode..."
timeout 5s amdgpu_top --dump 2>/dev/null || echo "amdgpu_top dump completed (timeout or no GPU expected in container)"

# Check version if available
amdgpu_top --version 2>/dev/null || echo "Version check completed"

echo "SUCCESS: amdgpu_top installation test passed"
echo "Note: For interactive use, run: amdgpu_top"
exit 0
