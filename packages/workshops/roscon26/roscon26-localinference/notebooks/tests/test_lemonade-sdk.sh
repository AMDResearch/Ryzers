#!/bin/bash
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

echo "Running tests for lemonade-sdk..."
echo "================================"

# Use the same image-baked model as notebooks 02-04.
MODEL="${LEMONADE_TEST_MODEL:-Gemma-4-E2B-it-GGUF}"
LEMONADE_CACHE="${LEMONADE_CACHE:-/opt/lemonade-cache/lemonade}"
LEMONADE_HF_HOME="${LEMONADE_HF_HOME:-/opt/lemonade-cache/huggingface}"
export HF_HOME="${LEMONADE_HF_HOME}"
# v10.x default lemond port
PORT=13305

# Start the lemond server in the background. Upstream v10.x split the CLI:
# `lemond` runs the server; `lemonade` is the client CLI (pull/list/etc).
echo ""
echo "Starting lemond on default port ${PORT}..."
lemond "${LEMONADE_CACHE}" > /tmp/lemonade.log 2>&1 &
SERVER_PID=$!

cleanup() {
  echo ""
  echo "Stopping server..."
  kill $SERVER_PID 2>/dev/null
  sleep 1
  pkill -9 lemond 2>/dev/null
}
trap cleanup EXIT

# Wait for server to be ready
echo "Waiting for server to be ready..."
for i in {1..60}; do
  if curl -s http://localhost:$PORT/api/v1/health > /dev/null 2>&1; then
    echo "Server is ready!"
    break
  fi
  if [ $i -eq 60 ]; then
    echo "Server failed to start!"
    cat /tmp/lemonade.log
    exit 1
  fi
  sleep 1
done

# Confirm the workshop model is baked, then load it without a network pull.
echo ""
echo "Loading image-cached model $MODEL..."
test -d "${LEMONADE_HF_HOME}/hub/models--unsloth--gemma-4-E2B-it-GGUF"
test -d "${LEMONADE_HF_HOME}/hub/models--unsloth--gemma-4-E4B-it-GGUF"
lemonade load "$MODEL"

sleep 2

# Test the API with an actual completion request
echo ""
echo "Testing completion API with model $MODEL..."
RESPONSE=$(curl -s -X POST http://localhost:$PORT/api/v1/completions   -H "Content-Type: application/json"   -d '{
    "model": "'$MODEL'",
    "prompt": "Why is the sky blue?",
    "max_tokens": 100,
    "temperature": 0.7
  }')

echo "Raw API Response:"
echo "$RESPONSE" | python3 -m json.tool || echo "$RESPONSE"

if echo "$RESPONSE" | grep -q "choices"; then
  echo ""
  echo "================================"
  echo "✓ lemonade-sdk API test passed!"
  exit 0
else
  echo ""
  echo "✗ API test failed - no valid completion received"
  exit 1
fi
