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

By default, the test uses `yolov11x.pt` - to use a different model, modify the `model_name` in [test_ultralytics.py](test_ultralytics.py).

## References

- [Ultralytics Documentation](https://docs.ultralytics.com/)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)

---

Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
