#!/usr/bin/env bash
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Extract AMDNPU firmware (.sbin) from linux-firmware.
# Usage: ./extract_npu_firmware.sh [target_dir]

set -euo pipefail

TARGET_DIR="${1:-/lib/firmware/amdnpu}"
WORK_DIR="$(mktemp -d)"
FW_GIT_URL="https://git.kernel.org/pub/scm/linux/kernel/git/firmware/linux-firmware.git/plain/amdnpu"

cleanup() { rm -rf "$WORK_DIR"; }
trap cleanup EXIT

# Try distro package first, then fall back to kernel.org git
extract_from_deb() {
    cd "$WORK_DIR"
    apt-get update -qq 2>/dev/null
    apt-get download linux-firmware 2>/dev/null || return 1
    mkdir -p extracted
    for deb in *.deb; do
        dpkg-deb --fsys-tarfile "$deb" \
            | tar -C extracted -x --wildcards '*/firmware/amdnpu/*' 2>/dev/null || true
    done
    local fw_src=""
    for d in extracted/lib/firmware/amdnpu extracted/usr/lib/firmware/amdnpu; do
        [ -d "$d" ] && fw_src="$d" && break
    done
    [ -z "$fw_src" ] && return 1
    cp -a "$fw_src"/* "$TARGET_DIR"/
}

extract_from_pacman() {
    cd "$WORK_DIR"
    # Arch linux-firmware or linux-firmware-other
    pacman -Sy --noconfirm linux-firmware 2>/dev/null || \
    pacman -Sy --noconfirm linux-firmware-other 2>/dev/null || return 1
    # Files are already installed to /usr/lib/firmware/amdnpu
    [ -d /usr/lib/firmware/amdnpu ] && cp -a /usr/lib/firmware/amdnpu/* "$TARGET_DIR"/ && return 0
    [ -d /lib/firmware/amdnpu ] && cp -a /lib/firmware/amdnpu/* "$TARGET_DIR"/ && return 0
    return 1
}

extract_from_git() {
    cd "$WORK_DIR"
    echo "Downloading firmware from kernel.org..."
    for dev_id in 1502_00 17f0_10 17f0_11 17f0_20; do
        mkdir -p "$TARGET_DIR/$dev_id"
        # Download whatever firmware files are listed for this device
        for fw in npu.sbin npu_7.sbin; do
            curl -fsSL "${FW_GIT_URL}/${dev_id}/${fw}" \
                -o "$TARGET_DIR/${dev_id}/${fw}" 2>/dev/null || true
        done
    done
}

echo "==> Installing firmware to $TARGET_DIR"
mkdir -p "$TARGET_DIR"

if command -v dpkg-deb &>/dev/null && command -v apt-get &>/dev/null; then
    echo "Using apt (Debian/Ubuntu)..."
    extract_from_deb || { echo "deb extraction failed, trying kernel.org..."; extract_from_git; }
elif command -v pacman &>/dev/null; then
    echo "Using pacman (Arch)..."
    # On Arch, firmware is already installed if linux-firmware is present
    if [ -d /usr/lib/firmware/amdnpu ]; then
        cp -a /usr/lib/firmware/amdnpu/* "$TARGET_DIR"/
    else
        extract_from_pacman || { echo "pacman failed, trying kernel.org..."; extract_from_git; }
    fi
else
    echo "No package manager found, downloading from kernel.org..."
    extract_from_git
fi

echo "==> Installed NPU firmware:"
find "$TARGET_DIR" -type f -o -type l 2>/dev/null | sort
echo "Done."
