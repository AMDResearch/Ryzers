#!/bin/bash

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT


timeout $AMDGPUTOP_TIMEOUT konsole --noclose -e amdgpu_top

echo "amdgpu_top test completed."
exit 0
