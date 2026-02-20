#!/usr/bin/env python3
# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.dataset import ModalityConfig
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy
import time
import torch

print("="*60)
print("GR00T Benchmark Test")
print("="*60)

# Get the data config
print("\nLoading data config...")
data_config = DATA_CONFIG_MAP["fourier_gr1_arms_only"]

# Get the modality configs and transforms
modality_config = data_config.modality_config()
transforms = data_config.transform()

# Load dataset
print("Loading dataset...")
dataset = LeRobotSingleDataset(
    dataset_path="demo_data/robot_sim.PickNPlace",
    modality_configs=modality_config,
    transforms=None,  # we can choose to not apply any transforms
    embodiment_tag=EmbodimentTag.GR1,  # the embodiment to use
)

# Load pre-trained model
print("Loading GR00T policy...")
policy = Gr00tPolicy(
    model_path="nvidia/GR00T-N1.5-3B",
    modality_config=modality_config,
    modality_transform=transforms,
    embodiment_tag=EmbodimentTag.GR1,
    device="cuda"
)
print("Policy loaded successfully")

# Get sample data
sample_data = dataset[0]

# Warmup
print("Warming up...")
for _ in range(3):
    _ = policy.get_action(sample_data)
torch.cuda.synchronize()

# Benchmark
print("Running benchmark...")
torch.cuda.reset_peak_memory_stats()
num_iterations = 100
latencies = []

for _ in range(num_iterations):
    torch.cuda.synchronize()
    start = time.perf_counter()
    action_chunk = policy.get_action(sample_data)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    latencies.append(elapsed * 1000)  # Convert to ms

avg_latency_ms = sum(latencies) / len(latencies)
std_latency = (sum((x - avg_latency_ms)**2 for x in latencies) / len(latencies)) ** 0.5
min_latency = min(latencies)
max_latency = max(latencies)
avg_latency_s = avg_latency_ms / 1000

# Get action horizon from output
action_horizon = len(action_chunk['action.left_arm'])
avg_hz = action_horizon / avg_latency_s

print(f"\n{'='*60}")
print("GR00T Results")
print(f"{'='*60}")
print(f"Action horizon: {action_horizon}")
print(f"Iterations: {num_iterations}")
print(f"Avg latency: {avg_latency_ms:.2f} ms ({avg_latency_s:.6f} s)")
print(f"Std deviation: {std_latency:.2f} ms")
print(f"Min latency: {min_latency:.2f} ms")
print(f"Max latency: {max_latency:.2f} ms")
print(f"Avg Hz: {avg_hz:.2f} Hz")
print(f"Max GPU memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
print(f"{'='*60}")
