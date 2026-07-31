#!/usr/bin/env bash
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Extract AMDNPU firmware (.sbin) from Ubuntu's linux-firmware deb.
# Usage: ./extract_npu_firmware.sh [target_dir]

set -euo pipefail

TARGET_DIR="${1:-/lib/firmware/amdnpu}"
WORK_DIR="$(mktemp -d)"

cleanup() { rm -rf "$WORK_DIR"; }
trap cleanup EXIT

echo "==> Downloading linux-firmware package..."
cd "$WORK_DIR"

# apt-get download fetches the deb without installing it.
# On Ubuntu 24.04+ the amdnpu firmware is part of linux-firmware.
apt-get update -qq
apt-get download linux-firmware 2>/dev/null \
    || apt-get download linux-firmware-amdgpu 2>/dev/null \
    || { echo "ERROR: could not download linux-firmware package"; exit 1; }

echo "==> Extracting AMDNPU firmware..."
mkdir -p extracted
for deb in *.deb; do
    dpkg-deb --fsys-tarfile "$deb" \
        | tar -C extracted -x --wildcards '*/firmware/amdnpu/*' 2>/dev/null || true
done

# Find where the firmware ended up (varies by package layout)
FW_SRC=""
for candidate in extracted/lib/firmware/amdnpu extracted/usr/lib/firmware/amdnpu; do
    if [ -d "$candidate" ]; then
        FW_SRC="$candidate"
        break
    fi
done

if [ -z "$FW_SRC" ]; then
    echo "ERROR: no amdnpu firmware found in downloaded package"
    echo "Available firmware dirs:"
    find extracted -type d -name 'firmware' 2>/dev/null
    exit 1
fi

echo "==> Installing firmware to $TARGET_DIR"
mkdir -p "$TARGET_DIR"
cp -a "$FW_SRC"/* "$TARGET_DIR"/

echo "==> Installed NPU firmware:"
find "$TARGET_DIR" -type f -o -type l | sort
echo "Done."
