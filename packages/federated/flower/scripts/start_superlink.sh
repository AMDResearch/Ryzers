#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

source $(dirname "$0")/env.sh

echo "Starting Flower Superlink..."

# Start the superlink container using ryzers run
# The container configuration (GPU, network, etc.) is defined in superlink/config.yaml
ryzers run --name superlink "--insecure --isolation process"

exec bash
