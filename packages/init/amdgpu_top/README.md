# AMD GPU Top - GPU Performance Monitoring

This Ryzer package installs and configures `amdgpu_top`, a command-line tool for monitoring AMD GPU performance and utilization in real-time. This tool provides detailed information about GPU memory usage, temperature, power consumption, and compute activity.

[GitHub] https://github.com/Umio-Yasuno/amdgpu_top

## Build and Run
```bash
ryzers build amdgpu_top
ryzers run
```

## Overview

`amdgpu_top` displays real-time statistics about:

- GPU utilization and frequency
- Memory usage (VRAM)
- Temperature and power consumption
- GPU engine activity (Graphics, Compute, Video Encode/Decode)
- Per-process GPU usage




## Configuration
`AMDGPUTOP_TIMEOUT=10s`: Default timeout for running amdgpu_top

Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.