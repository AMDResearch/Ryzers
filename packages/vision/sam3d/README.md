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
It uses Triton sparse convolutions, independent unmasked SDPA and per-stage
image-conditioning caches on both GPUs. Large FP16 sparse-latent attention blocks
use an architecture-tuned Triton kernel and fused normalization. Structure
sampling keeps PyTorch SDPA and reuses frozen linear-parameter casts. Inside the
container, `SAM3D_CACHE_CROSS_ATTN=0`, `SAM3D_TRITON_ATTN=0`,
`SAM3D_TRITON_NORM=0` and `SAM3D_CACHE_LINEAR=0` disable the respective
optimizations. The image enables experimental AOTriton so the iGPU uses fused SDPA.

Validated on Radeon 8060S (`gfx1151`) and Radeon AI PRO R9700 (`gfx1201`),
ROCm 7.14 / PyTorch 2.12. Final warm mask-14 medians were 65.01 s on 8060S
and 16.23 s on R9700, using default steps and both geometry decoders. See [VALIDATION.md](VALIDATION.md)
for numerical reference provenance, maximum errors, benchmark methods and
regression coverage.

Optional mesh cleanup, texture baking, layout optimization and Gaussian video
rendering need additional GPU rasterizers and are not supported here. The
upstream example disables these paths; PyTorch3D/gsplat are installed without
their CUDA rasterizers.

## References

- [Upstream setup](https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md)
- [Single-object notebook](https://github.com/facebookresearch/sam-3d-objects/blob/main/notebook/demo_single_object.ipynb)
- [Multi-object notebook](https://github.com/facebookresearch/sam-3d-objects/blob/main/notebook/demo_multi_object.ipynb)

Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
