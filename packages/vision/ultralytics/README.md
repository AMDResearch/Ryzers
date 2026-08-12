# Ultralytics YOLO

Ultralytics YOLO (You Only Look Once) is a state-of-the-art object detection and image segmentation model. This package includes support with AMD ROCm GPU acceleration.

## Build and Run

```bash
ryzers build ultralytics
ryzers run 
```

## Demo

To run the webcam demo, attach a webcam and run the ryzers run command below:

```bash
ryzers run "python3 /ryzers/demo_ultralytics.py"
```

## Models

The test and webcam demo use the bundled `yolo11n.pt` model. To use another model, update the path passed to `YOLO` in the corresponding script.

## References

- [Ultralytics Documentation](https://docs.ultralytics.com/)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)

---

Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
