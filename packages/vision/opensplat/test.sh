#!/bin/bash

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set -e

BUILD_DIR="/ryzers/OpenSplat/build"
TRAINER_BIN="${BUILD_DIR}/opensplat"

echo ">>> Starting OpenSplat test..."

echo ">>> Running '${TRAINER_BIN}'"
${TRAINER_BIN} /ryzers/data/banana -n 2000

echo ">>> OpenSplat test completed successfully."
