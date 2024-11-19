# OpenCV Docker Package

This package installs OpenCV as part of the Ryzer build flow.

## Usage

To include this package in your Ryzer build, add `opencv` to your list of packages in the `RyzerManager` configuration or `ryzer build` call.

Calling Directly from Shell Example:
```bash
OPENCV_VERSION=4.5.3 ryzers build opencv
docker run
```

Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.