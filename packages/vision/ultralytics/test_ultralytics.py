# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

from ultralytics import YOLO
import ultralytics
print(f"Ultralytics version: {ultralytics.__version__}")

print("Loading YOLO model...")
model = YOLO("/ryzers/data_ultralytics/yolo11x.pt")
print("Model loaded successfully")

print("Running inference benchmark (10 iterations)...")
for i in range(10):
    results = model("/ryzers/data_ultralytics/bus.jpg", verbose=False)
    if i == 0:
        # Print detection results from first run
        print(f"Detected {len(results[0].boxes)} objects in image")
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = model.names[cls_id]
            print(f"  - {cls_name}: {conf:.2f}")

# Save result image instead of displaying
output_path = "/ryzers/data_ultralytics/result.jpg"
results[0].save(output_path)
print(f"Result saved to: {output_path}")

print("SUCCESS: Ultralytics YOLO test passed")