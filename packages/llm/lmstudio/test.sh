#!/bin/bash

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set -e

PORT="${LMSTUDIO_PORT:-1234}"
API="http://127.0.0.1:${PORT}"

echo "Testing LM Studio installation..."

# The lms CLI drives llmster, the headless daemon. Both come from the installer.
if ! command -v lms &> /dev/null; then
    echo "FAIL: lms command not found"
    exit 1
fi

echo "lms found at: $(which lms)"

# Start the daemon. It detaches and stays up on its own, and this is what
# provisions ~/.lmstudio/.internal/lms-key-2 - without it every lms command
# below fails with ENOENT on that path.
echo "Starting the LM Studio daemon (llmster)..."
if ! lms daemon up; then
    echo "FAIL: could not start the llmster daemon"
    exit 1
fi

cleanup() {
    lms unload --all >/dev/null 2>&1 || true
    lms server stop >/dev/null 2>&1 || true
    lms daemon down >/dev/null 2>&1 || true
}
trap cleanup EXIT

lms daemon status

echo "Available models:"
lms ls || true

echo "Starting the OpenAI-compatible server on port ${PORT}..."
if ! lms server start --port "${PORT}"; then
    echo "FAIL: could not start the LM Studio server"
    exit 1
fi

if ! curl -sf -m 30 "${API}/v1/models" >/dev/null; then
    echo "FAIL: ${API}/v1/models did not respond"
    exit 1
fi

echo "Server responded at ${API}/v1/models"

# End-to-end generation, only when a model is already cached in the mounted
# models volume. A fresh checkout has none, and downloading one here would turn
# a sign-of-life test into a multi-GB download. Pick the smallest one to keep
# the test quick.
MODEL="$(lms ls --llm --json 2>/dev/null | python3 -c '
import json, sys
try:
    models = json.load(sys.stdin)
except Exception:
    sys.exit(0)
models = [m for m in models if m.get("modelKey")]
if models:
    print(min(models, key=lambda m: m.get("sizeBytes") or 0)["modelKey"])
' || true)"

if [[ -z "$MODEL" ]]; then
    echo "No local model found, skipping the generation test."
    echo "Download one with: lms get <model>"
else
    echo "Running a generation test with '${MODEL}'..."

    # The runtime survey reports what the daemon can offload to. An empty
    # accelerator list means inference silently ran on CPU.
    echo "Inference runtime:"
    survey="$(lms runtime survey 2>&1 || true)"
    echo "$survey"
    if ! grep -qi "vulkan\|rocm" <<<"$survey"; then
        echo "WARNING: no GPU accelerator detected, inference is running on CPU"
    fi

    # The server loads the model just-in-time on the first request.
    response="$(curl -sf -m 300 "${API}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word: pong\"}],\"max_tokens\":16,\"temperature\":0}")"

    lms ps || true
    lms unload --all >/dev/null 2>&1 || true

    if [[ -z "$response" ]]; then
        echo "FAIL: no response from ${API}/v1/chat/completions"
        exit 1
    fi

    echo "Model response: ${response}"
    if ! grep -q '"content"' <<<"$response"; then
        echo "FAIL: chat completion did not return any content"
        exit 1
    fi
fi

echo "SUCCESS: LM Studio installation test passed"
echo "Note: For headless use, run: lms daemon up && lms server start --port ${PORT}"
exit 0
