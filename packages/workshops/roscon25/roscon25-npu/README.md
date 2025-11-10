### ROSCon'25 AI PC Workshop: Perception with NPUs

This package contains the hands-on material from the workshop "Introducing AI PCs for Embodied AI" presented at [ROSCon 2025 Singapore](https://roscon.ros.org/2025/).

### Prerequisites

**Note: this package requires the XDNA driver to be installed on your host system.** Read the instructions on the [XDNA ryzer documentation](../../../npu/xdna) for installation help.

### Build and run the Docker Image

```sh
ryzers build xdna ryzenai_cvml ros roscon25-npu
ryzers run
```

### References

- https://github.com/amd/xdna-driver
- https://ryzenai.docs.amd.com/en/latest/ryzen_ai_libraries.html

Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
