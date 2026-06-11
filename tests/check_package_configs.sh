#!/bin/bash
# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Scans all packages and reports:
# - Whether package overrides the default init image
# - Whether package installs its own PyTorch

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PACKAGES_DIR="$REPO_ROOT/packages"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Counters
total_packages=0
image_overrides=0
pytorch_installs=0

# Arrays to store results
declare -a results

# Function to check if config.yaml has init_image override
check_config_image_override() {
    local config_file="$1"
    if [[ -f "$config_file" ]]; then
        # Look for uncommented init_image: line
        local init_image
        init_image=$(grep -E '^\s*init_image:\s*' "$config_file" 2>/dev/null | grep -v '^\s*#' | head -1)
        if [[ -n "$init_image" ]]; then
            # Extract the image name
            echo "$init_image" | sed 's/.*init_image:\s*["'"'"']\?\([^"'"'"']*\)["'"'"']\?.*/\1/'
            return 0
        fi
    fi
    return 1
}

# Function to check if Dockerfile has BASE_IMAGE with a default value
check_dockerfile_image_override() {
    local dockerfile="$1"
    # Look for ARG BASE_IMAGE=<something> (with a value, not just ARG BASE_IMAGE)
    local base_image_line
    base_image_line=$(grep -E '^ARG BASE_IMAGE=' "$dockerfile" 2>/dev/null | head -1)
    if [[ -n "$base_image_line" ]]; then
        # Extract the image name (after the =)
        local image_name
        image_name=$(echo "$base_image_line" | sed 's/ARG BASE_IMAGE=//')
        if [[ -n "$image_name" ]]; then
            echo "$image_name"
            return 0
        fi
    fi
    return 1
}

# Function to check if Dockerfile installs PyTorch
check_pytorch_install() {
    local dockerfile="$1"
    # Look for pip install patterns that include torch
    # Patterns: pip install.*torch, pip3 install.*torch, pip install --pre torch, etc.
    local pytorch_line
    pytorch_line=$(grep -E 'pip3?\s+install.*torch' "$dockerfile" 2>/dev/null | grep -v 'uninstall' | head -1)
    if [[ -n "$pytorch_line" ]]; then
        # Try to extract meaningful info about the install
        if echo "$pytorch_line" | grep -q 'rocm6\.4.*nightly\|nightly.*rocm6\.4'; then
            echo "rocm6.4 nightly"
        elif echo "$pytorch_line" | grep -q 'nightly'; then
            echo "nightly"
        elif echo "$pytorch_line" | grep -q 'rocm'; then
            # Extract rocm version if present
            local rocm_ver
            rocm_ver=$(echo "$pytorch_line" | grep -oE 'rocm[0-9]+\.[0-9]+' | head -1)
            if [[ -n "$rocm_ver" ]]; then
                echo "$rocm_ver"
            else
                echo "rocm"
            fi
        elif echo "$pytorch_line" | grep -q '\-\-pre'; then
            echo "pre-release"
        elif echo "$pytorch_line" | grep -q '\.whl'; then
            echo "wheel"
        elif echo "$pytorch_line" | grep -q '\-\-force-reinstall'; then
            echo "reinstall"
        else
            echo "yes"
        fi
        return 0
    fi
    return 1
}

# Function to truncate string with ellipsis
truncate_str() {
    local str="$1"
    local max_len="$2"
    if [[ ${#str} -gt $max_len ]]; then
        echo "${str:0:$((max_len-3))}..."
    else
        echo "$str"
    fi
}

# Print header
echo ""
echo "================================================================================="
echo "                     RYZERS PACKAGE CONFIGURATION REPORT"
echo "================================================================================="
echo ""
printf "%-20s %-12s %-28s %-20s\n" "PACKAGE" "CATEGORY" "IMAGE OVERRIDE" "PYTORCH INSTALL"
echo "-------------------------------------------------------------------------------"

# Find all packages (directories containing a Dockerfile under packages/)
while IFS= read -r dockerfile; do
    pkg_dir=$(dirname "$dockerfile")
    pkg_name=$(basename "$pkg_dir")

    # Extract category (parent directory of package)
    category=$(basename "$(dirname "$pkg_dir")")

    config_file="$pkg_dir/config.yaml"

    # Check for image override
    image_override=""
    image_source=""

    # First check config.yaml
    if config_image=$(check_config_image_override "$config_file"); then
        image_override="$config_image"
        image_source="config"
    fi

    # Then check Dockerfile (if no config override found)
    if [[ -z "$image_override" ]]; then
        if docker_image=$(check_dockerfile_image_override "$dockerfile"); then
            image_override="$docker_image"
            image_source="dockerfile"
        fi
    fi

    # Check for PyTorch installation
    pytorch_install=""
    if pytorch_info=$(check_pytorch_install "$dockerfile"); then
        pytorch_install="$pytorch_info"
    fi

    # Update counters
    ((total_packages++))
    [[ -n "$image_override" ]] && ((image_overrides++))
    [[ -n "$pytorch_install" ]] && ((pytorch_installs++))

    # Format output
    if [[ -n "$image_override" ]]; then
        image_display="YES ($(truncate_str "$image_override" 20))"
    else
        image_display="no"
    fi

    if [[ -n "$pytorch_install" ]]; then
        pytorch_display="YES ($pytorch_install)"
    else
        pytorch_display="no"
    fi

    # Print row
    printf "%-20s %-12s %-28s %-20s\n" "$pkg_name" "$category" "$image_display" "$pytorch_display"

done < <(find "$PACKAGES_DIR" -name "Dockerfile" -type f | sort)

# Print summary
echo ""
echo "================================================================================="
echo "SUMMARY"
echo "================================================================================="
echo "Total packages: $total_packages"
echo "Image overrides: $image_overrides"
echo "PyTorch installs: $pytorch_installs"
echo ""
