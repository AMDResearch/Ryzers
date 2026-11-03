# OpenSplat

[![CPU](https://img.shields.io/badge/ryzenai-x86-blue)](#)
[![GPU](https://img.shields.io/badge/ryzenai-gpu-blue)](#)

---

## Building and Running the Docker Container

To build and run the Docker container:

```bash
ryzers build opensplat
ryzers run
```
---

## Building an Example Splat

1. **Check `config.yaml`**:
   The default configuration mounts the host's `./data` directory at `/ryzers/data`. Update the mapping if your data is stored elsewhere.

2. **Generate the splat**:
   Start an interactive shell:
   ```bash
   ryzers run bash
   ```
   Then run:
   ```bash
   cd /ryzers/OpenSplat/build
   ./opensplat /opt/opensplat-test-data/banana -n 2000 \
       -o /ryzers/data/output_splat.ply
   ```
   This writes the result to `./data/output_splat.ply` on the host.

3. **View the splat**:
   Close the Docker container, then navigate to [https://playcanvas.com/viewer](https://playcanvas.com/viewer).  
   Drag and drop the `output_splat.ply` file into the viewer to see your generated splat.

---

## Further Details

For more information, visit the official [OpenSplat repository](https://github.com/pierotofy/OpenSplat/tree/main).

Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
