#!/usr/bin/env python3
# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from openpi.training import config as _config
from openpi.policies import policy_config
import numpy as np
import time
import torch
import os

# Disable autotuning
os.environ["HIPBLAS_AUTOTUNE"] = "0"

np.random.seed(42)

print("\nLoading PI0 policy...")
cfg = _config.get_config("pi0_droid")
ckpt = "/root/.cache/openpi/openpi-assets/checkpoints/pi0_droid"

policy = policy_config.create_trained_policy(cfg, ckpt)
print("Policy loaded successfully")

# Force disable compilation on the loaded model
if hasattr(policy, '_sample_actions') and hasattr(policy._sample_actions, '_torchdynamo_inline'):
    print("Detected compiled _sample_actions, replacing with eager version...")
    policy._sample_actions = policy._sample_actions.__wrapped__ if hasattr(policy._sample_actions, '__wrapped__') else policy._sample_actions

H, W = 224, 224
example = {
    "observation/exterior_image_1_left": np.zeros((H, W, 3), dtype=np.uint8),
    "observation/wrist_image_left":     np.zeros((H, W, 3), dtype=np.uint8),
    "observation/joint_position":       np.zeros((7,), dtype=np.float32),
    "observation/gripper_position":     np.array([0.0], dtype=np.float32),
    "prompt": "pick up the fork",
}

# Warmup
print("Warming up...")
_ = policy.infer(example)

# Benchmark
print("Running benchmark...")
torch.cuda.reset_peak_memory_stats()
num_iterations = 100
latencies = []

for _ in range(num_iterations):
    t0 = time.perf_counter()
    out = policy.infer(example)
    elapsed = time.perf_counter() - t0
    latencies.append(elapsed * 1000)  # Convert to ms

avg_latency_ms = np.mean(latencies)
std_latency = np.std(latencies)
min_latency = np.min(latencies)
max_latency = np.max(latencies)
avg_latency_s = avg_latency_ms / 1000

# Get action horizon from output
action_horizon = out['actions'].shape[1] if len(out['actions'].shape) > 1 else 1
avg_hz = action_horizon / avg_latency_s

print(f"\n{'='*60}")
print("PI0 Results")
print(f"{'='*60}")
print(f"Action horizon: {action_horizon}")
print(f"Iterations: {num_iterations}")
print(f"Avg latency: {avg_latency_ms:.2f} ms ({avg_latency_s:.6f} s)")
print(f"Std deviation: {std_latency:.2f} ms")
print(f"Min latency: {min_latency:.2f} ms")
print(f"Max latency: {max_latency:.2f} ms")
print(f"Avg Hz: {avg_hz:.2f} Hz")
print(f"Max GPU memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
print(f"Actions shape: {out['actions'].shape}")
print(f"{'='*60}")
