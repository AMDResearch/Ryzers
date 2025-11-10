#!/bin/bash

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set -e

echo "Starting tests for PYNQ..."

echo "Checking PYNQ import..."
python3 -c "import pynq; print(f'PYNQ version: {pynq.__version__}')"

echo "All tests passed successfully."
