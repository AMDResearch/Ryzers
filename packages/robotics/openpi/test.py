# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from openpi.training import config as _config
from openpi.policies import policy_config
import numpy as np
import time
import torch
import os
import shutil
import argparse

# Parse arguments first
parser = argparse.ArgumentParser(description='Test PI0 PyTorch inference')
parser.add_argument('--debug', action='store_true', help='Enable debug logging and tracing')
parser.add_argument('--clear-cache', action='store_true', help='Clear Triton cache before running')
args = parser.parse_args()

# Clear Triton cache if requested
if args.clear_cache:
    triton_cache = os.path.expanduser("~/.triton/cache")
    if os.path.exists(triton_cache):
        print(f"Clearing Triton cache: {triton_cache}")
        shutil.rmtree(triton_cache)

# Disable torch.compile completely
#os.environ["TORCH_COMPILE_DISABLE"] = "1"
#os.environ["TORCHDYNAMO_DISABLE"] = "1"

# Enable debugging only if requested
if args.debug:
    os.environ["HSA_ENABLE_SDMA"] = "0"
    os.environ["AMD_LOG_LEVEL"] = "4"
    os.environ["ROCBLAS_LAYER"] = "3"
    print("Debug mode enabled")

np.random.seed(42)

print("creating policy user")
cfg = _config.get_config("pi0_droid")
ckpt = "/root/.cache/openpi/openpi-assets/checkpoints/pi0_droid"

print(f"Loading checkpoint from {ckpt}...")
policy = policy_config.create_trained_policy(cfg, ckpt)

# Force disable compilation on the loaded model
if hasattr(policy, '_sample_actions') and hasattr(policy._sample_actions, '_torchdynamo_inline'):
    if args.debug:
        print("Detected compiled _sample_actions, replacing with eager version...")
    policy._sample_actions = policy._sample_actions.__wrapped__ if hasattr(policy._sample_actions, '__wrapped__') else policy._sample_actions

print("Policy loaded successfully")

if args.debug:
    print(f"\nGPU Memory before inference:")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

H, W = 224, 224
example = {
    "observation/exterior_image_1_left": np.zeros((H, W, 3), dtype=np.uint8),
    "observation/wrist_image_left":     np.zeros((H, W, 3), dtype=np.uint8),
    "observation/joint_position":       np.zeros((7,), dtype=np.float32),
    "observation/gripper_position":     np.array([0.0], dtype=np.float32),
    "prompt": "pick up the fork",
}

print("\nRunning inference...")
success = False
try:
    # Warmup run
    print("Warmup run...")
    _ = policy.infer(example)
    print("Warmup complete\n")
    
    # Timed runs
    num_iterations = 100
    latencies = []
    
    print(f"Running {num_iterations} iterations...")
    for i in range(num_iterations):
        t0 = time.perf_counter()
        out = policy.infer(example)
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed * 1000)  # Convert to ms
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{num_iterations}")
    
    # Calculate statistics
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    
    print(f"\n✓ Inference succeeded!")
    print(f"  Iterations: {num_iterations}")
    print(f"  Average latency: {avg_latency:.2f} ms")
    print(f"  Std deviation: {std_latency:.2f} ms")
    print(f"  Min latency: {min_latency:.2f} ms")
    print(f"  Max latency: {max_latency:.2f} ms")
    print(f"  Actions shape: {out['actions'].shape}")
    if args.debug:
        print(f"  Last policy timing: {out['policy_timing']['infer_ms']:.2f} ms")
    success = True
    
except RuntimeError as e:
    print(f"\n❌ Crash during inference!")
    print(f"Error: {e}")
    if args.debug:
        print(f"\nGPU Memory at crash:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    import traceback
    traceback.print_exc()

# Print autotuning results summary
if args.debug:
    try:
        from hipblas_autotuning import print_results
        print_results()
    except:
        print("\nAutotuning results not available")

# Final status
if success:
    print("\n" + "="*50)
    print("✓ TEST PASSED - All operations completed successfully")
    print("="*50)
    exit(0)
else:
    print("\n" + "="*50)
    print("❌ TEST FAILED - See errors above")
    print("="*50)
    exit(1)


