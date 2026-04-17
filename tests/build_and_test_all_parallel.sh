#!/bin/bash

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Build and test all Ryzers (parallel version)
# Usage: ./build_and_test_all_parallel.sh [--build-only] [--test-only] [--parallel N] [--skip PACKAGE] [--only PACKAGE]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$REPO_DIR/.venv"
LOG_DIR="$SCRIPT_DIR/build_logs"
RESULTS_DIR="$LOG_DIR/results"
RESULTS_FILE="$LOG_DIR/results_summary.txt"
LOCK_FILE="$LOG_DIR/.lock"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Associative arrays for dependency tracking
declare -A PACKAGE_DEPS      # package -> space-separated list of dependencies
declare -A FAILED_PACKAGES   # package -> 1 if failed

# Find all packages (directories containing a Dockerfile, excluding ryzer_env/ryzer_blank)
find_packages() {
    local packages=()
    while IFS= read -r dockerfile; do
        local pkg_dir=$(dirname "$dockerfile")
        local pkg_name=$(basename "$pkg_dir")
        # Skip init packages (ryzer_env, ryzer_blank) as they're base layers
        if [[ "$pkg_name" != "ryzer_env" && "$pkg_name" != "ryzer_blank" ]]; then
            packages+=("$pkg_name")
        fi
    done < <(find "$REPO_DIR/packages" -name "Dockerfile" -type f 2>/dev/null | sort)
    echo "${packages[@]}"
}

ALL_PACKAGES=($(find_packages))

# Build dependency map by scanning Dockerfiles for ARG BASE_IMAGE=<package>
build_dependency_map() {
    for pkg in "${ALL_PACKAGES[@]}"; do
        # Find the Dockerfile for this package
        local dockerfile
        dockerfile=$(find "$REPO_DIR/packages" -type d -name "$pkg" -exec find {} -maxdepth 1 -name "Dockerfile" \; 2>/dev/null | head -1)

        if [[ -f "$dockerfile" ]]; then
            # Look for ARG BASE_IMAGE=<value> where value is a simple name (not a full image path)
            local base_image
            base_image=$(grep -E '^ARG BASE_IMAGE=' "$dockerfile" 2>/dev/null | sed 's/ARG BASE_IMAGE=//' | head -1)

            # Check if base_image is a local package (not an external image like ubuntu:24.04 or rocm/pytorch:...)
            if [[ -n "$base_image" && ! "$base_image" =~ [/:] ]]; then
                # It's a simple name, check if it's one of our packages
                for other_pkg in "${ALL_PACKAGES[@]}"; do
                    if [[ "$base_image" == "$other_pkg" ]]; then
                        PACKAGE_DEPS["$pkg"]="$base_image"
                        break
                    fi
                done
                # Also check for init packages (xdna, etc.) that might not be in ALL_PACKAGES
                if [[ -z "${PACKAGE_DEPS[$pkg]}" && -d "$REPO_DIR/packages" ]]; then
                    # Search for the base image as a package directory
                    if find "$REPO_DIR/packages" -type d -name "$base_image" -exec test -f {}/Dockerfile \; -print -quit 2>/dev/null | grep -q .; then
                        PACKAGE_DEPS["$pkg"]="$base_image"
                    fi
                fi
            fi
        fi
    done
}

# Check if a package's dependencies have failed
check_deps_failed() {
    local pkg="$1"
    local dep="${PACKAGE_DEPS[$pkg]}"

    while [[ -n "$dep" ]]; do
        if [[ "${FAILED_PACKAGES[$dep]}" == "1" ]]; then
            return 0  # A dependency has failed
        fi
        # Check transitive dependencies
        dep="${PACKAGE_DEPS[$dep]}"
    done

    return 1  # No failed dependencies
}

# Get the dependency chain for a package (for display)
get_dep_chain() {
    local pkg="$1"
    local chain=""
    local dep="${PACKAGE_DEPS[$pkg]}"

    while [[ -n "$dep" ]]; do
        if [[ -n "$chain" ]]; then
            chain="$dep -> $chain"
        else
            chain="$dep"
        fi
        dep="${PACKAGE_DEPS[$dep]}"
    done

    echo "$chain"
}

# Default options
BUILD_ONLY=false
TEST_ONLY=false
PARALLEL=4
SKIP_PACKAGES=()
ONLY_PACKAGES=()
TIMEOUT=1800  # 30 minutes default timeout for tests
DRY_RUN=false
CLEAN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        --test-only)
            TEST_ONLY=true
            shift
            ;;
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        --skip)
            SKIP_PACKAGES+=("$2")
            shift 2
            ;;
        --only)
            ONLY_PACKAGES+=("$2")
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --list)
            echo "Available packages:"
            for pkg in "${ALL_PACKAGES[@]}"; do
                echo "  - $pkg"
            done
            exit 0
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Build and test all Ryzer packages (parallel version)."
            echo ""
            echo "Options:"
            echo "  --build-only     Only build, don't run tests"
            echo "  --test-only      Only run tests (assumes images are already built)"
            echo "  --parallel N     Run N builds in parallel (default: 4)"
            echo "  --skip PACKAGE   Skip specified package (can be used multiple times)"
            echo "  --only PACKAGE   Only build/test specified package (can be used multiple times)"
            echo "  --timeout SECS   Timeout for each test in seconds (default: 1800)"
            echo "  --dry-run        Print what would be done without executing"
            echo "  --clean          Delete all cached outputs (images, logs, run scripts)"
            echo "  --list           List all available packages"
            echo "  --help, -h       Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Setup logging directory
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

# Thread-safe logging with flock
log_message() {
    local message="$1"
    (
        flock -x 200
        echo -e "$message"
    ) 200>"$LOCK_FILE"
}

# Activate virtual environment
setup_venv() {
    if [[ ! -d "$VENV_DIR" ]]; then
        echo -e "${YELLOW}Creating virtual environment...${NC}"
        python3 -m venv "$VENV_DIR"
        source "$VENV_DIR/bin/activate"
        pip install -e "$REPO_DIR"
    else
        source "$VENV_DIR/bin/activate"
    fi
}

# Check if package should be processed
should_process() {
    local pkg="$1"

    # Check if in skip list
    for skip in "${SKIP_PACKAGES[@]}"; do
        if [[ "$pkg" == "$skip" ]]; then
            return 1
        fi
    done

    # Check if only list is specified
    if [[ ${#ONLY_PACKAGES[@]} -gt 0 ]]; then
        for only in "${ONLY_PACKAGES[@]}"; do
            if [[ "$pkg" == "$only" ]]; then
                return 0
            fi
        done
        return 1
    fi

    return 0
}

# Build a single package (writes result to per-package file)
build_package() {
    local pkg="$1"
    local log_file="$LOG_DIR/build_${pkg}.log"
    local result_file="$RESULTS_DIR/build_${pkg}.result"
    local start_time=$(date +%s)

    # Check if any dependency has failed
    if check_deps_failed "$pkg"; then
        local failed_dep
        for dep in ${PACKAGE_DEPS[$pkg]}; do
            if [[ "${FAILED_PACKAGES[$dep]}" == "1" ]]; then
                failed_dep="$dep"
                break
            fi
            # Check transitive deps
            local tdep="${PACKAGE_DEPS[$dep]}"
            while [[ -n "$tdep" ]]; do
                if [[ "${FAILED_PACKAGES[$tdep]}" == "1" ]]; then
                    failed_dep="$tdep"
                    break 2
                fi
                tdep="${PACKAGE_DEPS[$tdep]}"
            done
        done
        log_message "${YELLOW}[BUILD]${NC} $pkg - SKIPPED (dependency '$failed_dep' failed)"
        echo "BUILD,$pkg,DEP_FAILED,0" > "$result_file"
        FAILED_PACKAGES["$pkg"]=1  # Mark as failed so dependents also skip
        return 1
    fi

    log_message "${BLUE}[BUILD]${NC} Building $pkg..."

    if $DRY_RUN; then
        log_message "  DRY RUN: ryzers build --name ryzer-$pkg $pkg"
        echo "BUILD,$pkg,DRYRUN,0" > "$result_file"
        return 0
    fi

    if ryzers build --name "ryzer-$pkg" "$pkg" > "$log_file" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_message "${GREEN}[BUILD]${NC} $pkg - SUCCESS (${duration}s)"
        echo "BUILD,$pkg,SUCCESS,$duration" > "$result_file"
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_message "${RED}[BUILD]${NC} $pkg - FAILED (see $log_file)"
        echo "BUILD,$pkg,FAILED,$duration" > "$result_file"
        FAILED_PACKAGES["$pkg"]=1  # Mark as failed
        return 1
    fi
}

# Test a single package (writes result to per-package file)
test_package() {
    local pkg="$1"
    local image_name="ryzer-$pkg"
    local log_file="$LOG_DIR/test_${pkg}.log"
    local result_file="$RESULTS_DIR/test_${pkg}.result"
    local start_time=$(date +%s)

    log_message "${BLUE}[TEST]${NC} Testing $pkg..."

    if $DRY_RUN; then
        log_message "  DRY RUN: docker run --rm $image_name"
        echo "TEST,$pkg,DRYRUN,0" > "$result_file"
        return 0
    fi

    # Check if image exists
    if ! docker image inspect "$image_name" > /dev/null 2>&1; then
        log_message "${YELLOW}[TEST]${NC} $pkg - SKIPPED (image not found)"
        echo "TEST,$pkg,SKIPPED,0" > "$result_file"
        return 0
    fi

    # Run the container with timeout
    if timeout "$TIMEOUT" docker run --rm \
        --device=/dev/kfd --device=/dev/dri \
        --security-opt seccomp=unconfined \
        --group-add video --group-add render \
        --shm-size 16G \
        "$image_name" > "$log_file" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_message "${GREEN}[TEST]${NC} $pkg - SUCCESS (${duration}s)"
        echo "TEST,$pkg,SUCCESS,$duration" > "$result_file"
        return 0
    else
        local exit_code=$?
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        if [[ $exit_code -eq 124 ]]; then
            log_message "${YELLOW}[TEST]${NC} $pkg - TIMEOUT after ${TIMEOUT}s (see $log_file)"
            echo "TEST,$pkg,TIMEOUT,$duration" > "$result_file"
        else
            log_message "${RED}[TEST]${NC} $pkg - FAILED (see $log_file)"
            echo "TEST,$pkg,FAILED,$duration" > "$result_file"
        fi
        return 1
    fi
}

# Merge all per-package result files into summary
merge_results() {
    echo "TYPE,PACKAGE,STATUS,DURATION" > "$RESULTS_FILE"
    for result_file in "$RESULTS_DIR"/*.result; do
        if [[ -f "$result_file" ]]; then
            cat "$result_file" >> "$RESULTS_FILE"
        fi
    done
}

# Clean cached outputs
clean_all() {
    echo -e "${YELLOW}Cleaning cached outputs...${NC}"

    # Remove build logs
    if [[ -d "$LOG_DIR" ]]; then
        echo "  Removing build logs: $LOG_DIR"
        rm -rf "$LOG_DIR"
    fi

    # Remove generated run scripts and build logs in repo dir
    echo "  Removing generated run scripts and build logs..."
    rm -f "$REPO_DIR"/ryzers.run.*.sh
    rm -f "$REPO_DIR"/ryzers.build.*.log

    # Remove _ryzers.yaml tracking file
    if [[ -f "$REPO_DIR/ryzers/_ryzers.yaml" ]]; then
        echo "  Removing ryzers state file"
        rm -f "$REPO_DIR/ryzers/_ryzers.yaml"
    fi

    # Remove Docker images
    echo "  Removing Docker images..."
    for pkg in "${ALL_PACKAGES[@]}"; do
        local image_name="ryzer-$pkg"
        if docker image inspect "$image_name" > /dev/null 2>&1; then
            echo "    Removing image: $image_name"
            docker rmi "$image_name" 2>/dev/null || true
        fi
    done

    # Also remove the ryzer_env base image
    if docker image inspect "ryzer_env" > /dev/null 2>&1; then
        echo "    Removing image: ryzer_env"
        docker rmi "ryzer_env" 2>/dev/null || true
    fi

    echo -e "${GREEN}Clean complete.${NC}"
}

# Print summary
print_summary() {
    # Merge results first
    merge_results

    echo ""
    echo "=============================================="
    echo "                  SUMMARY"
    echo "=============================================="

    if [[ -f "$RESULTS_FILE" ]]; then
        local build_success=$(grep "^BUILD,.*,SUCCESS" "$RESULTS_FILE" | wc -l)
        local build_failed=$(grep "^BUILD,.*,FAILED" "$RESULTS_FILE" | wc -l)
        local build_dep_failed=$(grep "^BUILD,.*,DEP_FAILED" "$RESULTS_FILE" | wc -l)
        local test_success=$(grep "^TEST,.*,SUCCESS" "$RESULTS_FILE" | wc -l)
        local test_failed=$(grep "^TEST,.*,FAILED" "$RESULTS_FILE" | wc -l)
        local test_timeout=$(grep "^TEST,.*,TIMEOUT" "$RESULTS_FILE" | wc -l)
        local test_skipped=$(grep "^TEST,.*,SKIPPED" "$RESULTS_FILE" | wc -l)

        echo ""
        echo "Build Results:"
        echo -e "  ${GREEN}Success:${NC}     $build_success"
        echo -e "  ${RED}Failed:${NC}      $build_failed"
        echo -e "  ${YELLOW}Dep Failed:${NC}  $build_dep_failed"

        if ! $BUILD_ONLY; then
            echo ""
            echo "Test Results:"
            echo -e "  ${GREEN}Success:${NC} $test_success"
            echo -e "  ${RED}Failed:${NC}  $test_failed"
            echo -e "  ${YELLOW}Timeout:${NC} $test_timeout"
            echo -e "  ${YELLOW}Skipped:${NC} $test_skipped"
        fi

        echo ""
        echo "Failed builds:"
        grep "^BUILD,.*,FAILED" "$RESULTS_FILE" | cut -d',' -f2 | while read pkg; do
            echo -e "  ${RED}- $pkg${NC}"
        done

        if [[ $build_dep_failed -gt 0 ]]; then
            echo ""
            echo "Skipped due to dependency failure:"
            grep "^BUILD,.*,DEP_FAILED" "$RESULTS_FILE" | cut -d',' -f2 | while read pkg; do
                echo -e "  ${YELLOW}- $pkg${NC}"
            done
        fi

        if ! $BUILD_ONLY; then
            echo ""
            echo "Failed tests:"
            grep "^TEST,.*,FAILED" "$RESULTS_FILE" | cut -d',' -f2 | while read pkg; do
                echo -e "  ${RED}- $pkg${NC}"
            done
        fi

        echo ""
        echo "Full results: $RESULTS_FILE"
        echo "Build logs: $LOG_DIR/"
    fi
}

# Run jobs in parallel with limit (batch approach)
run_parallel() {
    local func="$1"
    shift
    local packages=("$@")
    local count=${#packages[@]}
    local i=0

    while [[ $i -lt $count ]]; do
        local pids=()
        local j=0

        # Start up to PARALLEL jobs
        while [[ $j -lt $PARALLEL && $i -lt $count ]]; do
            $func "${packages[$i]}" &
            pids+=($!)
            ((i++))
            ((j++))
        done

        # Wait for this batch to complete
        for pid in "${pids[@]}"; do
            wait "$pid" 2>/dev/null || true
        done
    done
}

# Topologically sort packages based on dependencies
# Packages with no dependencies come first, then packages depending on them, etc.
topo_sort_packages() {
    local -a sorted=()
    local -A visited=()
    local -a input_packages=("$@")

    # Helper function for DFS
    visit() {
        local pkg="$1"
        if [[ "${visited[$pkg]}" == "1" ]]; then
            return
        fi
        visited["$pkg"]=1

        # Visit dependencies first
        local dep="${PACKAGE_DEPS[$pkg]}"
        if [[ -n "$dep" ]]; then
            # Check if dep is in our input list
            for p in "${input_packages[@]}"; do
                if [[ "$p" == "$dep" ]]; then
                    visit "$dep"
                    break
                fi
            done
        fi

        sorted+=("$pkg")
    }

    for pkg in "${input_packages[@]}"; do
        visit "$pkg"
    done

    echo "${sorted[@]}"
}

# Main execution
main() {
    # Handle clean flag first
    if $CLEAN; then
        clean_all
        exit 0
    fi

    echo "=============================================="
    echo "   Ryzers Build and Test Script (Parallel)"
    echo "=============================================="
    echo ""
    echo "Script directory: $SCRIPT_DIR"
    echo "Log directory: $LOG_DIR"
    echo "Parallel jobs: $PARALLEL"
    echo "Build only: $BUILD_ONLY"
    echo "Test only: $TEST_ONLY"
    echo "Dry run: $DRY_RUN"
    echo ""

    # Clear previous results
    rm -rf "$RESULTS_DIR"
    mkdir -p "$RESULTS_DIR"

    # Build dependency map
    build_dependency_map

    # Show detected dependencies
    local has_deps=false
    for pkg in "${!PACKAGE_DEPS[@]}"; do
        if [[ -n "${PACKAGE_DEPS[$pkg]}" ]]; then
            if ! $has_deps; then
                echo "Package dependencies detected:"
                has_deps=true
            fi
            echo "  $pkg -> ${PACKAGE_DEPS[$pkg]}"
        fi
    done
    if $has_deps; then
        echo ""
    fi

    # Setup virtual environment
    setup_venv

    # Get list of packages to process
    packages_to_process=()
    for pkg in "${ALL_PACKAGES[@]}"; do
        if should_process "$pkg"; then
            packages_to_process+=("$pkg")
        fi
    done

    # Sort packages by dependency order
    packages_to_process=($(topo_sort_packages "${packages_to_process[@]}"))

    echo "Packages to process: ${#packages_to_process[@]}"
    echo ""

    # Build phase
    if ! $TEST_ONLY; then
        echo "=============================================="
        echo "              BUILD PHASE"
        echo "=============================================="
        echo ""

        if [[ ${#packages_to_process[@]} -gt 0 ]]; then
            # Separate packages into layers based on dependencies
            # Layer 0: packages with no deps or deps outside the build set
            # Layer 1: packages that depend on layer 0, etc.

            local -A pkg_layer=()
            local -a layers=()
            local max_layer=0

            # Compute layer for each package
            for pkg in "${packages_to_process[@]}"; do
                local dep="${PACKAGE_DEPS[$pkg]}"
                if [[ -z "$dep" ]]; then
                    pkg_layer["$pkg"]=0
                else
                    # Check if dep is in our build set
                    local dep_in_set=false
                    for p in "${packages_to_process[@]}"; do
                        if [[ "$p" == "$dep" ]]; then
                            dep_in_set=true
                            break
                        fi
                    done

                    if $dep_in_set; then
                        # We need to compute this after deps are computed
                        pkg_layer["$pkg"]=-1
                    else
                        pkg_layer["$pkg"]=0
                    fi
                fi
            done

            # Resolve layers iteratively
            local changed=true
            while $changed; do
                changed=false
                for pkg in "${packages_to_process[@]}"; do
                    if [[ "${pkg_layer[$pkg]}" == "-1" ]]; then
                        local dep="${PACKAGE_DEPS[$pkg]}"
                        if [[ -n "$dep" && "${pkg_layer[$dep]}" != "-1" ]]; then
                            pkg_layer["$pkg"]=$((${pkg_layer[$dep]} + 1))
                            if [[ ${pkg_layer[$pkg]} -gt $max_layer ]]; then
                                max_layer=${pkg_layer[$pkg]}
                            fi
                            changed=true
                        fi
                    fi
                done
            done

            # Build layer by layer
            for ((layer=0; layer<=max_layer; layer++)); do
                local layer_pkgs=()
                for pkg in "${packages_to_process[@]}"; do
                    if [[ "${pkg_layer[$pkg]}" == "$layer" ]]; then
                        layer_pkgs+=("$pkg")
                    fi
                done

                if [[ ${#layer_pkgs[@]} -gt 0 ]]; then
                    if [[ $layer -eq 0 && ${#layer_pkgs[@]} -gt 0 ]]; then
                        # Build first package of layer 0 synchronously to cache ryzer_env
                        echo -e "${YELLOW}Building first package to cache ryzer_env base layer...${NC}"
                        build_package "${layer_pkgs[0]}" || true

                        if [[ ${#layer_pkgs[@]} -gt 1 ]]; then
                            run_parallel build_package "${layer_pkgs[@]:1}"
                        fi
                    else
                        echo -e "${BLUE}Building layer $layer (${#layer_pkgs[@]} packages)...${NC}"
                        run_parallel build_package "${layer_pkgs[@]}"
                    fi
                fi
            done
        fi
    fi

    # Test phase
    if ! $BUILD_ONLY; then
        echo ""
        echo "=============================================="
        echo "               TEST PHASE"
        echo "=============================================="
        echo ""

        run_parallel test_package "${packages_to_process[@]}"
    fi

    # Print summary
    print_summary
}

# Run main
main
