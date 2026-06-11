#!/usr/bin/env bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set -e

echo "Testing MobileSAM installation..."

cd /ryzers/MobileSAM

# Download weights if not present
if [ ! -f weights/mobile_sam.pt ]; then
    echo "Downloading MobileSAM weights..."
    mkdir -p weights
    wget -q https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt -O weights/mobile_sam.pt
fi

# Run a simple inference test
echo "Running MobileSAM inference test..."
python3 << 'EOF'
import torch
from mobile_sam import sam_model_registry, SamPredictor
import numpy as np

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
print("Loading MobileSAM model...")
sam_checkpoint = "weights/mobile_sam.pt"
model_type = "vit_t"

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam = mobile_sam.to(device=device)
mobile_sam.eval()
print("Model loaded successfully")

# Create a dummy image and run inference
print("Running inference on test image...")
predictor = SamPredictor(mobile_sam)

# Create a simple test image (RGB, 256x256)
test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
predictor.set_image(test_image)

# Test point prompt
input_point = np.array([[128, 128]])
input_label = np.array([1])
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

print(f"Generated {len(masks)} masks")
print(f"Mask scores: {scores}")
print("SUCCESS: MobileSAM inference test passed!")
EOF

echo "MobileSAM test completed successfully"
