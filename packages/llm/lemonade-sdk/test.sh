#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

echo "Running tests for lemonade-sdk..."
echo "================================"

# Use a small model for fast testing
MODEL="Llama-3.2-1B-Instruct-GGUF"

# Pull the model first
echo ""
echo "Pulling model $MODEL..."
lemonade-server pull $MODEL || {
  echo "Model already available or pull failed, continuing..."
}

# List available models
echo ""
echo "Available models:"
lemonade-server list | head -20

# Start lemonade-server in the background
echo ""
echo "Starting lemonade-server..."
lemonade-server serve --no-tray > /tmp/lemonade.log 2>&1 &
SERVER_PID=$!

# Function to cleanup
cleanup() {
  echo ""
  echo "Stopping server..."
  kill $SERVER_PID 2>/dev/null
  sleep 1
  pkill -9 lemonade-server 2>/dev/null
}

# Set trap to cleanup on exit
trap cleanup EXIT

# Wait for server to be ready
echo "Waiting for server to be ready..."
for i in {1..60}; do
  if curl -s http://localhost:8000/health > /dev/null 2>&1; then
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

# Give it a moment to fully initialize
sleep 2

# Test the API with an actual completion request
echo ""
echo "Testing completion API with model $MODEL..."
echo "Prompt: 'Why is the sky blue? Answer in one sentence.'"
echo ""

RESPONSE=$(curl -s -X POST http://localhost:8000/api/v1/completions   -H "Content-Type: application/json"   -d '{
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
