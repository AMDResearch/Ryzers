# SAM 3D Objects

Reconstruct an object's mesh, appearance and placement from an image and mask
using Meta's [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects)
on ROCm PyTorch.

## Build and run

```sh
ryzers build sam3d
ryzers run
```

The default is a headless smoke test. It checks GPU access, imports and a small
sparse convolution against a CPU reference, without downloading model weights.

## Reconstruct the example

Accept the [model license](https://huggingface.co/facebook/sam-3d-objects) and
export a Hugging Face token with access to the model:

```sh
read -rsp 'Hugging Face token: ' HF_TOKEN; echo
export HF_TOKEN
ryzers run "python /ryzers/test_sam3d.py"
```

This reconstructs the stuffed bear in Meta's example room (mask 14). To select
other masks or assemble the complete room:

```sh
ryzers run "python /ryzers/test_sam3d.py --mask-index 14 3"
ryzers run "python /ryzers/test_sam3d.py --all-masks"
```

GLB meshes with vertex colors and PLY Gaussian splats are saved in
`workspace/sam3d`. Model weights and compiled kernels are cached under
`workspace/.cache`. The first reconstruction downloads weights and compiles
kernels. Meta recommends at least 32 GB of GPU memory.

For systems with multiple GPUs, list the available devices and select one:

```sh
ryzers run "python -c t=__import__('torch');print(list(enumerate(map(t.cuda.get_device_name,range(t.cuda.device_count())))))"
HIP_VISIBLE_DEVICES=0 ryzers run "python /ryzers/test_sam3d.py"  # use the chosen index
```

Device indices can change after a reboot. `HF_TOKEN` and `HIP_VISIBLE_DEVICES`
are forwarded at runtime by the package config.

## ROCm support

Tested with ROCm 7.14 / PyTorch 2.12 on the Ryzen AI Max+ 395's integrated
Radeon 8060S (`gfx1151`) and Radeon AI PRO R9700 (`gfx1201`). The package uses
Triton sparse convolutions and PyTorch SDPA. Architecture-specific attention,
normalization and linear optimizations have guards and retain PyTorch paths
for other architectures, input shapes and execution modes.

The image enables experimental AOTriton attention for `gfx1151` and uses the
native PyTorch allocator to avoid first-pass depth failures with expandable
segments. On `gfx1151`, MIOpen solver selection defaults to `FAST`; an explicit
`MIOPEN_FIND_MODE` takes precedence. This is a process-wide runtime setting.

Mesh reconstruction, scene assembly and GLB/PLY export are supported. Meta's
Gaussian video renderer and optional mesh cleanup, texture baking and layout
refinement need additional GPU rasterizers. PyTorch3D and gsplat are built
without their NVIDIA GPU extensions.

## Development checks

Additional checks for the custom kernels and inference caches require no model
weights:

```sh
ryzers run "python /ryzers/test_sam3d_conditioning_cache.py"
ryzers run "python /ryzers/check_sam3d_numerics.py"
```

Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
