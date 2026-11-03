#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Smoke test: verify the flower-supernode binary is installed and the
# --help output renders. We do NOT attempt to connect to a SuperLink in
# the test, since that would require a live federation.

set -e

echo "Running tests for flower-supernode..."

flower-supernode --help >/dev/null

# Sanity-check that the expected flags are documented
flower-supernode --help 2>&1 | grep -q -- "--superlink" \
  || { echo "FAIL: --superlink flag missing from help"; exit 1; }
flower-supernode --help 2>&1 | grep -q -- "--clientappio-api-address" \
  || { echo "FAIL: --clientappio-api-address flag missing from help"; exit 1; }

echo "flower-supernode CLI looks good."
echo "Tests passed!"
