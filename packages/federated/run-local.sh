#!/bin/bash
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# One-shot single-machine smoke test for the Flower federated-learning
# Ryzers. Brings up SuperLink + ServerApp superexec + 2×SuperNode +
# 2×ClientApp superexec on the host network (127.0.0.1 + offset ports),
# submits the quickstart-pytorch run, and tears everything down.
#
# Multi-machine deployment uses the same three Ryzers — just point
# SUPERLINK_IP at the server and run the appropriate role on each box.
# See each Ryzer's README for the distributed flow.
#
# Usage:
#   cd packages/federated
#   ./run-local.sh                    # default 2 partitions
#   FLOWER_NUM_PARTITIONS=4 ./run-local.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

NUM_PARTITIONS="${FLOWER_NUM_PARTITIONS:-2}"

cd "${REPO_ROOT}"

echo "== Building Ryzers =="
ryzers build flower-base flower-superlink
ryzers build flower-base flower-supernode
ryzers build flower-base flower-superexec

echo "== Cleaning up stale containers =="
for name in flower-superlink flower-supernode flower-superexec; do
  docker ps -aq --filter "ancestor=${name}" | xargs -r docker rm -f >/dev/null 2>&1 || true
done

# All background PIDs and per-instance log files so we can clean up on exit.
BG_PIDS=()
LOG_DIR="$(mktemp -d -t flower-local-XXXXXX)"
echo "Logs: ${LOG_DIR}"

cleanup() {
  echo "== Tearing down =="
  for pid in "${BG_PIDS[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
  # --rm in each config.yaml means containers self-clean on exit,
  # but kill any stragglers just in case.
  for name in flower-superlink flower-supernode flower-superexec; do
    docker ps -q --filter "ancestor=${name}" | xargs -r docker kill >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT INT TERM

wait_for_port() {
  local port="$1"
  local timeout="${2:-30}"
  for _ in $(seq 1 "${timeout}"); do
    if (echo > /dev/tcp/127.0.0.1/"${port}") 2>/dev/null; then
      return 0
    fi
    sleep 1
  done
  echo "Timed out waiting for 127.0.0.1:${port}" >&2
  return 1
}

echo "== Starting SuperLink =="
ryzers run --name flower-superlink >"${LOG_DIR}/superlink.log" 2>&1 &
BG_PIDS+=($!)
wait_for_port 9092 30   # FleetApi — last of SuperLink's ports to bind

echo "== Starting ServerApp superexec =="
FLOWER_PLUGIN_TYPE=serverapp \
FLOWER_APPIO_ADDR=127.0.0.1:9091 \
  ryzers run --name flower-superexec >"${LOG_DIR}/serverapp.log" 2>&1 &
BG_PIDS+=($!)

for i in $(seq 0 $((NUM_PARTITIONS - 1))); do
  echo "== Starting SuperNode partition ${i} =="
  FLOWER_PARTITION_ID="${i}" \
  FLOWER_NUM_PARTITIONS="${NUM_PARTITIONS}" \
    ryzers run --name flower-supernode >"${LOG_DIR}/supernode-${i}.log" 2>&1 &
  BG_PIDS+=($!)

  echo "== Starting ClientApp superexec partition ${i} =="
  FLOWER_PLUGIN_TYPE=clientapp \
  FLOWER_PARTITION_ID="${i}" \
  FLOWER_APPIO_ADDR="127.0.0.1:$((9094 + i))" \
    ryzers run --name flower-superexec >"${LOG_DIR}/clientapp-${i}.log" 2>&1 &
  BG_PIDS+=($!)
done

echo "== Waiting for SuperLink ExecApi (9093) =="
wait_for_port 9093 30

# Give the supernodes a beat to register with the SuperLink before
# submitting — otherwise `flwr run` can race the fleet handshake.
sleep 5

echo "== Submitting quickstart-pytorch run =="
FLOWER_PLUGIN_TYPE=submit ryzers run --name flower-superexec | tee "${LOG_DIR}/submit.log"

echo
echo "== Done. Per-component logs: ${LOG_DIR} =="
