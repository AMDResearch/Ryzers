#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

echo "Checking camera usage on your system..."
echo "-------------------------------------"

# Loop through all /dev/video* devices
for cam in /dev/video*; do
    # Skip if no devices exist
    [ -e "$cam" ] || continue

    if lsof "$cam" &>/dev/null; then
        # List the process IDs and names using the camera
        echo "$cam is IN USE by:"
        lsof "$cam" | awk 'NR>1 {print "  PID: "$2", Process: "$1}'
    else
        echo "$cam is free"
    fi
done
