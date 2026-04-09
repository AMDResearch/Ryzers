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

# Step 1: Build containers
log_info "Building flower-superlink..."
ryzers build flower-superlink
log_success "flower-superlink built"

log_info "Building flower-supernode..."
ryzers build flower-supernode
log_success "flower-supernode built"

log_info "Building flower-superexec..."
ryzers build flower-superexec
log_success "flower-superexec built"

# Step 2: Create bridge network
log_info "Creating Docker bridge network: $NETWORK_NAME"
docker network rm "$NETWORK_NAME" 2>/dev/null || true
docker network create --driver bridge "$NETWORK_NAME"
log_success "Network created"

# Step 3: Start SuperLink
log_info "Starting SuperLink..."
docker run --rm -d \
    --network="$NETWORK_NAME" \
    --name=superlink \
    -p 9091:9091 -p 9092:9092 -p 9093:9093 \
    ryzers:flower-superlink \
    flower-superlink --insecure --isolation process
log_success "SuperLink started (ports 9091-9093)"

# Wait for SuperLink to be ready
sleep 2

# Step 4: Start SuperNodes
log_info "Starting SuperNode 1 (partition 0/2)..."
docker run --rm -d \
    --network="$NETWORK_NAME" \
    --name=supernode-1 \
    -p 9094:9094 \
    ryzers:flower-supernode \
    flower-supernode --insecure \
    --superlink superlink:9092 \
    --node-config "partition-id=0 num-partitions=2" \
    --clientappio-api-address 0.0.0.0:9094 \
    --isolation process
log_success "SuperNode 1 started (port 9094)"

log_info "Starting SuperNode 2 (partition 1/2)..."
docker run --rm -d \
    --network="$NETWORK_NAME" \
    --name=supernode-2 \
    -p 9095:9095 \
    ryzers:flower-supernode \
    flower-supernode --insecure \
    --superlink superlink:9092 \
    --node-config "partition-id=1 num-partitions=2" \
    --clientappio-api-address 0.0.0.0:9095 \
    --isolation process
log_success "SuperNode 2 started (port 9095)"

# Wait for SuperNodes to connect
sleep 2

# Step 5: Start SuperExec containers
log_info "Starting SuperExec (ServerApp)..."
docker run --rm -d \
    --network="$NETWORK_NAME" \
    --name=superexec-server \
    ryzers:flower-superexec \
    flower-superexec --insecure \
    --executor-type serverapp \
    --executor-config 'superlink="superlink:9091"'
log_success "SuperExec ServerApp started"

log_info "Starting SuperExec (ClientApp 1)..."
docker run --rm -d \
    --network="$NETWORK_NAME" \
    --name=superexec-client-1 \
    --device=/dev/kfd --device=/dev/dri \
    --security-opt seccomp=unconfined \
    --group-add video --group-add render \
    -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
    ryzers:flower-superexec \
    flower-superexec --insecure \
    --executor-type clientapp \
    --executor-config 'supernode="supernode-1:9094"'
log_success "SuperExec ClientApp 1 started"

log_info "Starting SuperExec (ClientApp 2)..."
docker run --rm -d \
    --network="$NETWORK_NAME" \
    --name=superexec-client-2 \
    --device=/dev/kfd --device=/dev/dri \
    --security-opt seccomp=unconfined \
    --group-add video --group-add render \
    -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
    ryzers:flower-superexec \
    flower-superexec --insecure \
    --executor-type clientapp \
    --executor-config 'supernode="supernode-2:9095"'
log_success "SuperExec ClientApp 2 started"

# Wait for all components to be ready
sleep 3

echo ""
log_info "All containers running:"
docker ps --filter "network=$NETWORK_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ""

# Step 6: Run federated learning
log_info "Starting federated learning..."
echo ""
echo "========================================"
echo "  Training CNN on CIFAR-10"
echo "  2 clients, 3 rounds, FedAvg"
echo "========================================"
echo ""

# Run flwr from the quickstart directory
cd "$SCRIPT_DIR/quickstart"

# Configure flwr to connect to our SuperLink
mkdir -p ~/.flwr
cat > ~/.flwr/config.toml << EOF
[superlink.local-deployment]
address = "127.0.0.1:9093"
insecure = true
EOF

# Run the federated learning job
flwr run . local-deployment --stream

echo ""
echo "========================================"
log_success "Federated learning complete!"
echo "========================================"
echo ""

# Show container logs summary
log_info "Container status:"
docker ps --filter "network=$NETWORK_NAME" --format "table {{.Names}}\t{{.Status}}"

echo ""
log_info "To view logs: docker logs <container-name>"
log_info "Containers will be cleaned up on exit (Ctrl+C)"
echo ""

# Keep running until user exits
read -p "Press Enter to cleanup and exit..."
