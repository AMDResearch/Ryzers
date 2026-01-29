# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from ultralytics import YOLO
import cv2

model = YOLO("/ryzers/data_ultralytics/yolo11x.pt")  # or s/m/l/x

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

frame_count = 0

print("========================================================")
print("Demo Starting ... Press 'q' in the opencv window to quit")
print("========================================================")

while True:
    ret, frame = cap.read()

    # ---- Guard 1: camera read failure ----
    if not ret or frame is None or frame.size == 0:
        print("Skipping empty frame")
        continue

    # Run inference
    results = model(frame, verbose=False)

    # ---- Guard 2: model returned nothing ----
    if not results or len(results) == 0:
        cv2.imshow("YOLOv11 Webcam", frame)
        continue

    result = results[0]

    # ---- Guard 3: no detections ----
    if result.boxes is None or len(result.boxes) == 0:
        annotated_frame = frame  # show original
    else:
        annotated_frame = result.plot()  # draw boxes

    cv2.imshow("YOLOv11 Webcam", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Processed {frame_count} frames safely.")