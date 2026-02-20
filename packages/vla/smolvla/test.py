#!/usr/bin/env python3
# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import torch
import time
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
from transformers import AutoProcessor

# Load model
print("\nLoading SmolVLA model...")
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base", revision="3326b10").to("cuda")
policy.eval()

# patch: The loaded policy is missing the language_tokenizer attribute.
policy.language_tokenizer = AutoProcessor.from_pretrained(policy.config.vlm_model_name).tokenizer

# Dummy batch config for a single observation
batch_size = 1
img_shape = (3, 512, 512)  # (C, H, W)
# Infer state_dim from the loaded normalization stats
state_dim = policy.normalize_inputs.buffer_observation_state.mean.shape[-1]

dummy_batch = {
    # a single image observation
    "observation.image": torch.rand(batch_size, *img_shape, device="cuda"),
    # a single state observation
    "observation.state": torch.rand(batch_size, state_dim, device="cuda"),
    "task": ["stack the blocks"] * batch_size,
}

# Prepare inputs for the model
normalized_batch = policy.normalize_inputs(dummy_batch)
images, img_masks = policy.prepare_images(normalized_batch)
state = policy.prepare_state(normalized_batch)
lang_tokens, lang_masks = policy.prepare_language(normalized_batch)

# Warmup
print("Warming up...")
for _ in range(3):
    with torch.no_grad():
        output = policy.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state)
torch.cuda.synchronize()

# Benchmark
print("Running benchmark...")
torch.cuda.reset_peak_memory_stats()
num_iterations = 100
latencies = []

for _ in range(num_iterations):
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        _ = policy.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    latencies.append(elapsed * 1000)  # Convert to ms

avg_latency_ms = sum(latencies) / len(latencies)
avg_latency_s = avg_latency_ms / 1000
avg_hz = policy.config.chunk_size / avg_latency_s

print(f"\n{'='*60}")
print("SmolVLA Results")
print(f"{'='*60}")
print(f"Chunk size: {policy.config.chunk_size}")
print(f"Iterations: {num_iterations}")
print(f"Avg latency: {avg_latency_ms:.2f} ms ({avg_latency_s:.6f} s)")
print(f"Avg Hz: {avg_hz:.2f} Hz")
print(f"Max GPU memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
print(f"{'='*60}")
