#!/bin/bash

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set -e

echo "Starting tests for segment-anything..."

echo "Checking module import..."
python3 -c "import segment_anything; print('segment_anything imported successfully!')"
python3 /ryzers/test_sam.py

echo "All tests passed successfully."