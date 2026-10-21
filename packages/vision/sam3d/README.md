# SAM 3D Objects

Reconstruct an object's 3D geometry, vertex colors, pose and Gaussian splat from
an image and segmentation mask using Meta's [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects).
This package uses ROCm PyTorch, PyTorch SDPA and Triton sparse convolutions.
It targets both Ryzen AI integrated graphics and discrete Radeon GPUs.

## Build and run

Accept the [SAM 3D Objects model license](https://huggingface.co/facebook/sam-3d-objects)
and export a Hugging Face token with access to that repository:

```sh
read -rsp 'Hugging Face token: ' HF_TOKEN; echo
export HF_TOKEN
ryzers build --name sam3d sam3d
ryzers run
```

The default test runs the upstream `demo.py` example: kidsroom image, mask 14,
seed 42, both inference stages, mesh and Gaussian decoding, without `torch.compile`.
Weights download on first run and persist in `workspace/.cache/huggingface`
and `workspace/.cache/torch`. Triton kernels compile and tune on first use; their
cache persists in `workspace/.cache/triton`.
PLY splats, vertex-colored GLB meshes and JSON validation reports persist in
`workspace/sam3d`.
The token is passed at runtime using Docker's `--env HF_TOKEN`; do not put it in
`config.yaml` or build arguments.

On a machine with multiple GPUs, select one with sufficient memory before running:

```sh
# Indices on the validation machine; choose the appropriate index on your system.
HIP_VISIBLE_DEVICES=0 ryzers run  # Ryzen AI integrated GPU
HIP_VISIBLE_DEVICES=1 ryzers run  # discrete Radeon GPU
```

The configuration forwards `HIP_VISIBLE_DEVICES` into Docker. Upstream recommends
at least 32 GB GPU memory.

For an installation check without model downloads, or another upstream mask:

```sh
ryzers run "python /ryzers/test_sam3d.py --smoke"
ryzers run "python /ryzers/test_sam3d.py --mask-index 14 3"
# Run all kidsroom masks and assemble the posed scene, as in the multi-object notebook.
ryzers run "python /ryzers/test_sam3d.py --all-masks"
```

## ROCm adaptations

The Dockerfile pins upstream SAM 3D and installs only inference dependencies,
keeping the base ROCm PyTorch installation. `patch_rocm.py` redirects spconv
imports to [spconv-triton](https://github.com/L-Reichardt/spconv-triton), removes
unused Kaolin notebook imports, and replaces the FlexiCubes shape-check helper
with equivalent torch tensor checks. FlexiCubes reconstruction stays on the GPU.

The image enables `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` before importing
PyTorch so gfx1151 uses fused SDPA. Without it, PyTorch selects the much slower,
quadratic-memory math backend on the iGPU.

Sparse attention runs each independent sequence/window through unmasked SDPA.
This preserves the original block-diagonal attention semantics while avoiding
the large additive mask and allowing fused kernels. The original implementation
remains available as `reference_masked_sdpa` in the upstream attention module.

Both GPUs also reuse the image-conditioning K/V projections during each
denoising stage. Conditional and unconditional inputs have separate entries;
the cache checks tensor/weight mutation counters, stores at most 3 GiB of outputs per stage,
and clears on return or exception. Training and gradient-enabled calls retain
upstream behavior. Tensors without mutation counters use the uncached path.
Set `SAM3D_CACHE_CROSS_ATTN=0` inside the container to compare with the uncached
implementation. Cache lifecycle checks require no GPU or model download:

```sh
ryzers run "python /ryzers/test_sam3d_conditioning_cache.py"
```

The default upstream example disables mesh cleanup, texture baking and layout
optimization. Those optional paths and Gaussian video rendering require
additional GPU rasterizers and are not supported by this package. PyTorch3D is
built with CPU extensions for its transforms and structures; gsplat is installed
without its CUDA extension for upstream import compatibility.

## Validation

Tested with ROCm 7.14 and PyTorch 2.12 on both GPUs, using the upstream
kidsroom image, seed 42, default inference steps and both geometry decoders.
Output counts below are representative; fresh-process reconstructions can vary
slightly in occupancy and mesh topology:

| GPU | Mask 14 Gaussians | GLB vertices | GLB faces |
|-----|------------------|--------------|-----------|
| Ryzen AI Radeon 8060S (`gfx1151`) | 842,880 | 489,102 | 978,208 |
| Radeon AI PRO R9700 (`gfx1201`) | 843,072 | 490,110 | 980,272 |

The R9700 also passed all 27 upstream masks and `make_scene` assembly,
producing a posed scene with 13,611,264 Gaussians. The 8060S passed masks 14
and 3 and assembled a 1,401,408-Gaussian scene. PLY and GLB files are
reloaded and checked for nonempty, finite geometry; all PLY attributes are
validated (opacity logits are decoded before checking). The smoke test
compares sparse convolution against a dense PyTorch reference on each GPU.

For an independent attention check against CPU float64 softmax/matmul, run:

```sh
ryzers run "python /ryzers/check_sam3d_numerics.py"
```

The sparse backend can also be checked against frozen fixtures generated with
the original CUDA spconv implementation. Use the same trusted, pinned checkout
as the Dockerfile; the checker verifies fixture hashes against its manifest:

```sh
ryzers run "bash"
# Inside the container:
git clone https://github.com/L-Reichardt/spconv-triton.git /tmp/spconv-reference
git -C /tmp/spconv-reference checkout 63361695e891de7aff32841ebb86f2661a41b059
python /ryzers/check_sam3d_numerics.py --golden-root /tmp/spconv-reference
```

This checks 39 forward cases covering submanifold, strided, dilated, transposed
and inverse 3D convolution, multiple batches, empty batches, channel tails and
float16/float32 with channels up to 256. Float16 uses the tight fixture family
whose bounds derive from CUDA float16 rounding against CUDA float32 references.
Reports persist at `/ryzers/output/numerics.json`. These operator checks do not
establish end-to-end agreement with a full CUDA SAM 3D run.

Warm mask-14 inference improved from **127.3 to 86.5 seconds on the 8060S**
and **25.1 to 20.7 seconds on the R9700** after the attention change. These are
medians of two timed runs after warming each implementation, excluding model
loading and export. Pose and sparse-structure occupancy were unchanged on each
device; latent-feature RMS differences were about 0.024%, comparable to repeating
the baseline. See [the numerical and performance report](VALIDATION.md) for
methodology, error bounds and remaining optimization targets.

The default conditioning cache provides a further **1.105x speedup on 8060S**
(88.14 -> 79.75 s) and **1.050x on R9700** (21.31 -> 20.30 s), measured in a
separate alternating comparison with three warm runs per mode. Both optimizations
are enabled by default on both GPUs. Cache comparisons preserved occupancy and
pose exactly; maximum latent differences and sampled geometry distances are
recorded in the validation report. Peak allocated tensor memory stayed at 20.3 GB
for mask 14.

## References

- [Upstream setup](https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md)
- [Single-object notebook](https://github.com/facebookresearch/sam-3d-objects/blob/main/notebook/demo_single_object.ipynb)
- [Multi-object notebook](https://github.com/facebookresearch/sam-3d-objects/blob/main/notebook/demo_multi_object.ipynb)

Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
