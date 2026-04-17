# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Example adapted from: https://huggingface.co/facebook/sam3

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

from transformers import Sam3Processor, Sam3Model
from PIL import Image
import numpy as np
import matplotlib

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading SAM3 model...")
model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")
print("Model loaded successfully")

# Load image
print("Loading test image...")
image = Image.open("/ryzers/data/toucan.jpg").convert("RGB")
print(f"Image size: {image.size}")

# Segment using text prompt
print("Running segmentation with text prompt 'bird'...")
inputs = processor(images=image, text="bird", return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

# Post-process results
results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.5,
    mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist()
)[0]

print(f"Found {len(results['masks'])} objects")
# Results contain:
# - masks: Binary masks resized to original image size
# - boxes: Bounding boxes in absolute pixel coordinates (xyxy format)
# - scores: Confidence scores

for i, (mask, score) in enumerate(zip(results['masks'], results['scores'])):
    print(f"  Object {i+1}: score={score:.3f}, mask shape={mask.shape}")

# Function to overlay masks on the image
def overlay_masks(image, masks):
    image = image.convert("RGBA")
    masks = 255 * masks.cpu().numpy().astype(np.uint8)
    n_masks = masks.shape[0]
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(n_masks)]
    for mask, color in zip(masks, colors):
        mask = Image.fromarray(mask)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)
    return image

# Save result image instead of displaying via NiceGUI
print("Generating segmentation overlay...")
image_with_mask = overlay_masks(image, results["masks"])
output_path = "/ryzers/data/sam3_result.png"
image_with_mask.save(output_path)
print(f"Result saved to: {output_path}")

print("SUCCESS: SAM3 segmentation test passed")