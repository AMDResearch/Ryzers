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

1. **Download the data**:  
   Create a directory called `data`. Download the “banana” data from the [official repository](https://github.com/pierotofy/OpenSplat/tree/main) and place it in `data`.

2. **Edit `config.yaml`**:  
   In your `config.yaml`, uncomment the relevant mount line. Update the local path is correct for mounting the `data` folder, as it will also be used to store your final splat.

3. **Generate the splat**:  
   Once inside the Docker container, run:
   ```bash
   cd build
   ./opensplat /ryzers/data/banana -n 2000
   ```
   This will generate the `output_splat.ply` by default.

4. **View the splat**:  
   Close the Docker container, then navigate to [https://playcanvas.com/viewer](https://playcanvas.com/viewer).  
   Drag and drop the `output_splat.ply` file into the viewer to see your generated splat.

---

## Further Details

For more information, visit the official [OpenSplat repository](https://github.com/pierotofy/OpenSplat/tree/main).

Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
