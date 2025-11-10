#!/usr/bin/env python3
# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import torch
import time
import os
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# Model and image
model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
image_path = "/ryzers/data/toucan.jpg"

print(f"Loading model: {model_name}")
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, device_map="auto")

print(f"Loading image: {image_path}")
image = Image.open(image_path).convert("RGB")

# Prepare inputs
inputs = processor(images=image, return_tensors="pt").to(model.device)

# Warmup
print("Warming up...")
with torch.inference_mode():
    for _ in range(3):
        _ = model(**inputs)
        torch.cuda.synchronize()

# Benchmark
print("Running benchmark...")
torch.cuda.reset_peak_memory_stats()
start = time.time()
runs = 10
for _ in range(runs):
    with torch.inference_mode():
        outputs = model(**inputs)
        torch.cuda.synchronize()
end = time.time()

avg_time = (end - start) / runs
print(f"\nAvg inference time: {avg_time:.6f} s ({1/avg_time:.2f} fps)")
print(f"Max GPU memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
print(f"Pooled output shape: {outputs.pooler_output.shape}")
print(f"Hidden state shape: {outputs.last_hidden_state.shape}")

print("Test done!")
