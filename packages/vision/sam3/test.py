# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Example adapted from: https://huggingface.co/facebook/sam3

from transformers import Sam3Processor, Sam3Model
import torch
from PIL import Image
import numpy as np
import matplotlib
from nicegui import ui
import base64
import io

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")

# Load image
image = Image.open("/ryzers/data/toucan.jpg").convert("RGB")

# Segment using text prompt
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

image_with_mask = overlay_masks(image, results["masks"])

# Convert PIL images to base64 data URIs for NiceGUI
def pil_to_base64(pil_image):
    buf = io.BytesIO()
    pil_image.save(buf, format='PNG')
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"

# Create NiceGUI page to display results
@ui.page('/')
def index():
    ui.label('SAM3 segmentation results').classes('text-2xl font-bold mb-4')
    with ui.row().classes('w-full gap-4'):
        with ui.column().classes('w-[45%]'):
            ui.label('Original image')
            ui.image(pil_to_base64(image.convert("RGB"))).classes('w-full')
        with ui.column().classes('w-[45%]'):
            ui.label('Segmentation result')
            ui.image(pil_to_base64(image_with_mask.convert("RGB"))).classes('w-full')

ui.run()