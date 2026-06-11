#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

echo "Running tests for lemonade-sdk..."
echo "================================"

# Use a small model for fast testing
MODEL="Llama-3.2-1B-Instruct-GGUF"
# v10.x default lemond port
PORT=13305

# Start the lemond server in the background. Upstream v10.x split the CLI:
# `lemond` runs the server
# `lemonade` is the client CLI for everything else (pull/list/etc).
echo ""
echo "Starting lemond on default port ${PORT}..."
lemond > /tmp/lemonade.log 2>&1 &
SERVER_PID=$!

# Function to cleanup
cleanup() {
  echo ""
  echo "Stopping server..."
  kill $SERVER_PID 2>/dev/null
  sleep 1
  pkill -9 lemond 2>/dev/null
}

# Set trap to cleanup on exit
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
    echo "Server logs:"
    cat /tmp/lemonade.log
    exit 1
  fi
  sleep 1
done

# Pull the model (the client talks to the server now that it's up)
echo ""
echo "Pulling model $MODEL..."
lemonade pull $MODEL || {
  echo "Pull failed; the server may still be able to lazy-load on first request."
}

# List available models
echo ""
echo "Available models (downloaded):"
lemonade list --downloaded | head -20

# Give it a moment to fully initialize
sleep 2

# Test the API with an actual completion request
echo ""
echo "Testing completion API with model $MODEL..."
echo "Prompt: 'Why is the sky blue? Answer in one sentence.'"
echo ""

RESPONSE=$(curl -s -X POST http://localhost:$PORT/api/v1/completions   -H "Content-Type: application/json"   -d '{
    "model": "'$MODEL'",
    "prompt": "Why is the sky blue?",
    "max_tokens": 100,
    "temperature": 0.7
  }')

echo "Raw API Response:"
echo "$RESPONSE" | python3 -m json.tool || echo "$RESPONSE"

# Check if we got a valid response
if echo "$RESPONSE" | grep -q "choices"; then
  echo ""
  echo "================================"
  echo "Generated Text:"
  echo "$RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['choices'][0]['text'] if data.get('choices') else 'No text generated')" 2>/dev/null || echo "Could not parse response"
  
  echo ""
  echo "Stats:"
  echo "$RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); usage=data.get('usage', {}); timings=data.get('timings', {}); print(f\"Prompt tokens: {usage.get('prompt_tokens', 'N/A')}\"); print(f\"Completion tokens: {usage.get('completion_tokens', 'N/A')}\"); print(f\"Total tokens: {usage.get('total_tokens', 'N/A')}\"); print(f\"Inference speed: {timings.get('predicted_per_second', 'N/A'):.1f} tokens/sec\" if timings.get('predicted_per_second') else '')" 2>/dev/null || echo "Stats not available"
  
  echo ""
  echo "================================"
  echo "✓ API test passed!"
  echo ""
  echo "✓ All tests passed!"
  exit 0
else
  echo ""
  echo "✗ API test failed - no valid completion received"
  echo ""
  echo "✗ Tests failed!"
  exit 1
fi
