#!/usr/bin/env python3
#
# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
import time

device = "cuda"

# Load Processor & VLA
print("\nLoading OpenVLA model...")
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    attn_implementation="sdpa",  # options are eager, sdpa, flash_attention_2
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device)
print("Model loaded successfully")

# Grab image input & format prompt
image_path = "/ryzers/data/toucan.jpg"
image = Image.open(image_path).convert("RGB")
prompt = "In: What action should the robot take to turn left?"
inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)

# Warmup
print("Warming up...")
for _ in range(3):
    with torch.no_grad():
        _ = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
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
        action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    latencies.append(elapsed * 1000)  # Convert to ms

avg_latency_ms = sum(latencies) / len(latencies)
std_latency = (sum((x - avg_latency_ms)**2 for x in latencies) / len(latencies)) ** 0.5
min_latency = min(latencies)
max_latency = max(latencies)
avg_latency_s = avg_latency_ms / 1000

# Get action dimension from output
action_dim = len(action)
avg_hz = 1 / avg_latency_s

print(f"\n{'='*60}")
print("OpenVLA Results")
print(f"{'='*60}")
print(f"Action dimension: {action_dim}")
print(f"Iterations: {num_iterations}")
print(f"Avg latency: {avg_latency_ms:.2f} ms ({avg_latency_s:.6f} s)")
print(f"Std deviation: {std_latency:.2f} ms")
print(f"Min latency: {min_latency:.2f} ms")
print(f"Max latency: {max_latency:.2f} ms")
print(f"Avg Hz: {avg_hz:.2f} Hz")
print(f"Max GPU memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
print(f"{'='*60}")
