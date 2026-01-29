# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from ultralytics import YOLO

model =  YOLO("/ryzers/data_ultralytics/yolo11x.pt")

for i in range(10):
    results = model("/ryzers/data_ultralytics/bus.jpg")

results[0].show()