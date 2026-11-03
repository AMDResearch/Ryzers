#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Smoke test: launch the SuperLink in --insecure mode in the background,
# confirm it binds the Fleet API port (9092), then shut it down. Intended
# to validate the install, NOT to run a real deployment.

set -e

echo "Running tests for flower-superlink..."

flower-superlink --help >/dev/null

flower-superlink --insecure &
PID=$!
trap "kill $PID 2>/dev/null || true" EXIT

# Wait up to 15s for port 9092 to be listening
for i in $(seq 1 15); do
  if (echo >/dev/tcp/127.0.0.1/9092) >/dev/null 2>&1; then
    echo "SuperLink is listening on 9092"
    echo "Tests passed!"
    exit 0
  fi
  sleep 1
done

echo "FAIL: SuperLink did not bind port 9092 within 15s"
exit 1
