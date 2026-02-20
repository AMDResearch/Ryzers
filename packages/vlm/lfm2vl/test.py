#!/usr/bin/env python3
#
# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Test code reference from: https://huggingface.co/LiquidAI/LFM2-VL-1.6B

from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.image_utils import load_image

# Load model and processor
model_id = "LiquidAI/LFM2-VL-1.6B"
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    device_map="auto",
    dtype="bfloat16"
)
processor = AutoProcessor.from_pretrained(model_id)

# Load image and create conversation
url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = load_image(url)
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "What is in this image?"},
        ],
    },
]

# Generate Answer
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    tokenize=True,
).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=64)
decoded_output = processor.batch_decode(outputs, skip_special_tokens=True)[0]
print(decoded_output)
