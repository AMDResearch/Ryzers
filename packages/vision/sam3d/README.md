# SAM 3D Objects

Reconstruct an object's mesh, vertex colors, pose and Gaussian splat from an image
and segmentation mask using Meta's [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects).
Runs on Ryzen AI integrated graphics and discrete Radeon GPUs using ROCm PyTorch.

## Build and run

```sh
ryzers build sam3d
ryzers run
```

The default runs a headless smoke test: imports SAM 3D, checks the ROCm GPU and
compares 1x1 sparse convolution with a dense CPU PyTorch reference. It exits on
completion and requires no model download or Hugging Face token. First
reconstruction compiles Triton kernels; the kernel cache persists in
`workspace/.cache/triton`.

## Reconstruct the example

Accept the [model license](https://huggingface.co/facebook/sam-3d-objects) and
export a Hugging Face token with access to that repository:

```sh
read -rsp 'Hugging Face token: ' HF_TOKEN; echo
export HF_TOKEN
ryzers run "python /ryzers/test_sam3d.py"
```

This runs the upstream kidsroom example, mask 14, seed 42, default inference
steps and both geometry decoders, without `torch.compile`. Upstream recommends
at least 32 GB GPU memory. Weights download on first reconstruction and persist
in `workspace/.cache/huggingface` and `workspace/.cache/torch`. Exported PLY splats,
vertex-colored GLB meshes and validation JSON persist in `workspace/sam3d`.
The token is forwarded at runtime; keep it out of `config.yaml` and build arguments.

Select a GPU on machines with multiple devices (indices below are specific to
the validation machine):

```sh
HIP_VISIBLE_DEVICES=0 ryzers run "python /ryzers/test_sam3d.py"  # 8060S iGPU
HIP_VISIBLE_DEVICES=1 ryzers run "python /ryzers/test_sam3d.py"  # R9700
```

Additional masks and multi-object scene assembly are explicit checks:

```sh
ryzers run "python /ryzers/test_sam3d.py --mask-index 14 3"
ryzers run "python /ryzers/test_sam3d.py --all-masks"
```

## ROCm support and validation

The image pins upstream sources and retains the base ROCm PyTorch installation.
Both validated GPUs use Triton sparse convolutions, independent unmasked SDPA,
bounded conditioning caches and architecture-tuned FP16 sparse-latent attention.
The default also compacts constant zero-conditioning attention, avoids Q/K/V
repacking and uses the native GEMM/bias/GELU epilogue for eligible latent MLPs.
Structure sampling preserves intermediate rounding; its RMS normalization tail
is fused on R9700 and retains PyTorch on 8060S. The iGPU reuses a faster MLP
weight layout.

Four DINO preprocessing wrappers share one byte-identical frozen backbone;
matching backbone features are reused across stages. Frozen structure-sampler
linear weights use the FP16 storage already used by autocast, while normalization
parameters remain FP32. Expandable allocator segments and a shared NumPy buffer
for PLY export reduce temporary allocations, including RAM shared with the iGPU.
These optimizations are enabled by default, with guards for supported shapes,
precision and inference mode. See [VALIDATION.md](VALIDATION.md) for overrides.

Validated on Radeon 8060S (`gfx1151`) and Radeon AI PRO R9700 (`gfx1201`),
ROCm 7.14 / PyTorch 2.12. Warm mask-14 medians were **50.59 s on 8060S** and
**12.99 s on R9700**, at default steps with both geometry decoders. Both devices
exported all 27 example objects and assembled the complete scene. Peak allocated
GPU memory during the full scene runs was **14.74 GiB** on either device;
checkpoint-loading peaks are excluded. See [VALIDATION.md](VALIDATION.md) for
reference provenance, maximum errors, coverage and measurement methods.

Optional mesh cleanup, texture baking, layout optimization and Gaussian video
rendering need additional GPU rasterizers and are not supported here. The
upstream example disables these paths; PyTorch3D/gsplat are installed without
their CUDA rasterizers.

## References

- [Upstream setup](https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md)
- [Single-object notebook](https://github.com/facebookresearch/sam-3d-objects/blob/main/notebook/demo_single_object.ipynb)
- [Multi-object notebook](https://github.com/facebookresearch/sam-3d-objects/blob/main/notebook/demo_multi_object.ipynb)

Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
