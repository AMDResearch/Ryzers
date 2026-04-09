#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Flower Federated Learning Demo
# Runs the complete PyTorch CIFAR-10 quickstart with 2 federated clients

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RYZERS_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
NETWORK_NAME="flwr-network"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

cleanup() {
    log_info "Cleaning up containers..."
    docker stop superlink supernode-1 supernode-2 superexec-server superexec-client-1 superexec-client-2 2>/dev/null || true
    docker rm superlink supernode-1 supernode-2 superexec-server superexec-client-1 superexec-client-2 2>/dev/null || true
    log_info "Removing network..."
    docker network rm "$NETWORK_NAME" 2>/dev/null || true
    log_success "Cleanup complete"
}

# Handle Ctrl+C
trap cleanup EXIT

echo "========================================"
echo "  Flower Federated Learning Demo"
echo "  PyTorch CIFAR-10 on AMD GPUs"
echo "========================================"
echo ""

# Check if ryzers is available
if ! command -v ryzers &> /dev/null; then
    log_error "ryzers command not found. Please install ryzers first."
    log_info "Run: pip install -e $RYZERS_DIR"
    exit 1
fi

cd "$RYZERS_DIR"

# Step 1: Build containers using ryzers build with explicit image names
log_info "Building flower-superlink..."
ryzers build --name flower-superlink flower-superlink
log_success "flower-superlink built"

log_info "Building flower-supernode..."
ryzers build --name flower-supernode flower-supernode
log_success "flower-supernode built"

log_info "Building flower-superexec..."
ryzers build --name flower-superexec flower-superexec
log_success "flower-superexec built"

# Step 2: Create bridge network for container-to-container communication
log_info "Creating Docker bridge network: $NETWORK_NAME"
docker network rm "$NETWORK_NAME" 2>/dev/null || true
docker network create --driver bridge "$NETWORK_NAME"
log_success "Network created"

# Step 3: Start SuperLink
# ryzers run uses the generated script, we pass instance overrides via the script's $1 and $2
log_info "Starting SuperLink..."
bash ryzers.run.flower-superlink.sh "--name=superlink --network=$NETWORK_NAME -d" "flower-superlink --insecure --isolation process"
log_success "SuperLink started (ports 9091-9093)"

# Wait for SuperLink to be ready
sleep 2

# Step 4: Start SuperNodes
log_info "Starting SuperNode 1 (partition 0/2)..."
bash ryzers.run.flower-supernode.sh "--name=supernode-1 --network=$NETWORK_NAME -p 9094:9094 -d" "flower-supernode --insecure --superlink superlink:9092 --node-config 'partition-id=0 num-partitions=2' --clientappio-api-address 0.0.0.0:9094 --isolation process"
log_success "SuperNode 1 started (port 9094)"

log_info "Starting SuperNode 2 (partition 1/2)..."
bash ryzers.run.flower-supernode.sh "--name=supernode-2 --network=$NETWORK_NAME -p 9095:9095 -d" "flower-supernode --insecure --superlink superlink:9092 --node-config 'partition-id=1 num-partitions=2' --clientappio-api-address 0.0.0.0:9095 --isolation process"
log_success "SuperNode 2 started (port 9095)"

# Wait for SuperNodes to connect
sleep 2

# Step 5: Start SuperExec containers
log_info "Starting SuperExec (ServerApp)..."
bash ryzers.run.flower-superexec.sh "--name=superexec-server --network=$NETWORK_NAME -d" "flower-superexec --insecure --executor flwr.superexec.deployment:executor --executor-config superlink=\"superlink:9091\""
log_success "SuperExec ServerApp started"

log_info "Starting SuperExec (ClientApp 1)..."
bash ryzers.run.flower-superexec.sh "--name=superexec-client-1 --network=$NETWORK_NAME -d" "flower-superexec --insecure --executor flwr.superexec.deployment:executor --executor-config superlink=\"supernode-1:9094\""
log_success "SuperExec ClientApp 1 started"

log_info "Starting SuperExec (ClientApp 2)..."
bash ryzers.run.flower-superexec.sh "--name=superexec-client-2 --network=$NETWORK_NAME -d" "flower-superexec --insecure --executor flwr.superexec.deployment:executor --executor-config superlink=\"supernode-2:9095\""
log_success "SuperExec ClientApp 2 started"

# Wait for all components to be ready
sleep 3

echo ""
log_info "All containers running:"
docker ps --filter "network=$NETWORK_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ""

# Step 6: Verify connectivity
log_info "Verifying container connectivity..."

echo -n "  SuperLink (9093): "
if docker run --rm --network="$NETWORK_NAME" busybox nc -z superlink 9093 2>/dev/null; then
    echo "OK"
else
    echo "FAILED"
fi

echo -n "  SuperNode-1 (9094): "
if docker run --rm --network="$NETWORK_NAME" busybox nc -z supernode-1 9094 2>/dev/null; then
    echo "OK"
else
    echo "FAILED"
fi

echo -n "  SuperNode-2 (9095): "
if docker run --rm --network="$NETWORK_NAME" busybox nc -z supernode-2 9095 2>/dev/null; then
    echo "OK"
else
    echo "FAILED"
fi

echo ""

# Show container logs for any errors
log_info "Recent container logs (last 3 lines each):"
for container in superlink supernode-1 supernode-2 superexec-server superexec-client-1 superexec-client-2; do
    echo "--- $container ---"
    docker logs --tail 3 "$container" 2>&1 || echo "(not running)"
done
echo ""

# Step 7: Run federated learning
log_info "Starting federated learning..."
echo ""
echo "========================================"
echo "  Training CNN on CIFAR-10"
echo "  2 clients, 3 rounds, FedAvg"
echo "========================================"
echo ""

# Run flwr using the generated script (interactive, not detached)
log_info "Submitting federated learning job..."
bash ryzers.run.flower-superexec.sh "--network=$NETWORK_NAME" '/bin/bash -c "mkdir -p ~/.flwr && echo -e \"[superlink.local-deployment]\naddress = \\\"superlink:9093\\\"\ninsecure = true\" > ~/.flwr/config.toml && cd /ryzers/quickstart && flwr run . local-deployment --stream"'

echo ""
echo "========================================"
log_success "Federated learning complete!"
echo "========================================"
echo ""

log_info "To view logs: docker logs <container-name>"
log_info "Containers will be cleaned up on exit (Ctrl+C)"
echo ""

read -p "Press Enter to cleanup and exit..."
