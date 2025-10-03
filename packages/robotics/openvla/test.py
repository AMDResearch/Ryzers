#!/usr/bin/env python3
#
# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

import torch

device = "cuda"

# Load Processor & VLA
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    attn_implementation="sdpa",  # options are eager, sdpa, flash_attention_2
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device)

# Grab image input & format prompt
image_path = "/ryzers/data/toucan.jpg"
print(f"Loading image from {image_path}")
image = Image.open(image_path).convert("RGB")

prompt = "In: What action should the robot take to turn left?"
print(f"Prompt: {prompt}")

# Predict Action (7-DoF; un-normalize for BridgeData V2)
inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

# Execute...
print(f"Action: {action}")
