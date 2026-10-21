# Numerical and performance validation

Validated 2026-10-21 on Radeon 8060S (`gfx1151`) and Radeon AI PRO R9700
(`gfx1201`), ROCm 7.14 / PyTorch 2.12. SAM 3D source is pinned to
`f91db411c50efee93d8db7aeb323885650f6f722`; model weights are
`facebook/sam-3d-objects` revision `2e73555018d2741ccd486e56c24fac41155a1dc6`.

## Reproduce the checks

`ryzers run` performs a headless installation smoke test without weights or
credentials. The following explicit checks require the model access described
in the README only when they reconstruct images:

```sh
ryzers run "python /ryzers/check_sam3d_numerics.py"
ryzers run "python /ryzers/test_sam3d_conditioning_cache.py"
ryzers run "python /ryzers/test_sam3d.py --mask-index 14 3"
ryzers run "python /ryzers/test_sam3d.py --all-masks"
```

For frozen CUDA sparse-convolution references, use a trusted pinned checkout:

```sh
ryzers run bash
# Inside the container:
git clone https://github.com/L-Reichardt/spconv-triton.git /tmp/spconv-reference
git -C /tmp/spconv-reference checkout 63361695e891de7aff32841ebb86f2661a41b059
python /ryzers/check_sam3d_numerics.py --golden-root /tmp/spconv-reference
```

The checker verifies fixture SHA-256 hashes before loading their pickled
metadata. Reports persist in `workspace/sam3d/numerics.json`. Run each check on
both devices using `HIP_VISIBLE_DEVICES` as shown in the README.

## Numerical references and maximum errors

The rebuilt default image passed all checks on both devices. The 39 sparse
fixtures come from the pinned spconv-triton checkout, whose reference generators
use unchanged CUDA spconv. Coordinates and spatial shapes match exactly;
features use the fixtures' original absolute tolerances with relative tolerance
zero. The seven tight FP16 cases compare against CUDA FP32 values with bounds
derived from CUDA's own FP16 rounding; older loose FP16 fixtures are excluded.
Coverage includes tails, up to 256 channels, empty/multiple batches, submanifold,
strided, dilated, transposed and inverse convolution.

| Check and reference | R9700 maximum absolute error | 8060S maximum absolute error |
|---|---:|---:|
| 32 FP32 sparse CUDA fixtures | 8.5831e-6 | 8.5831e-6 |
| 7 FP16 sparse CUDA fixtures | 9.7609e-4 | 9.7943e-4 |
| FP32 attention, CPU float64 | 4.9863e-7 | 4.9863e-7 |
| FP16 attention, CPU float64 | 6.2741e-4 | 6.2741e-4 |
| BF16 attention, CPU float64 | 4.4183e-3 | 4.4183e-3 |
| FP16 Triton attention, sampled CPU float64 | 6.2402e-5 | 1.0694e-4 |
| Six synthetic SLAT normalization cases, CPU float64 | 1.9505e-3 | 1.9456e-3 |
| Native FP16 GEMM/bias/tanh-GELU, sampled CPU float64 | 9.7065e-4 | 9.5104e-4 |
| Compact zero context, PyTorch reference | 0 | 0 |
| Twenty pointwise cases, PyTorch reference | 0 | 0 |
| Cached linear casts, PyTorch reference | 0 | 0 |

Attention checks cover unequal lengths, independent blocks, empty query blocks,
packed strides and tails. Absolute tolerances are 2e-5 / 2e-3 / 2e-2 for
FP32 / FP16 / BF16, with relative tolerance zero. Triton checks sample 32 query
rows against every key/value at lengths 2,049/2,071 and 4,096/4,096.
Normalization, layout and GEMM/GELU float64 checks use mixed
`atol=rtol=0.002`; this is not a universal 0.002 absolute-error bound.

Additional real-checkpoint operator replays sampled 32 rows per shape. FP16
attention at 5,252 queries and 5,252/5,496 keys reached maximum absolute error
0.001043 / 0.001054 on R9700 / 8060S against CPU float64. Sparse RMS reached
0.003441 / 0.003852 and SLAT layer normalization 0.003829 / 0.003614. A
2,048-channel decoder-shaped normalization replay reached 0.02963 / 0.02992
on larger-magnitude outputs and passed the same mixed tolerance; actual decoder
normalization retains PyTorch. The iGPU MLP layout check differed from row-major
PyTorch by at most 0.000244140625, and from float64 by 0.000122829.

Sixteen dense RMS cases match PyTorch numerically exactly on R9700, including
zero/subnormal inputs and packed strides; 8060S retains the PyTorch path.
Eight compact-context cases cover parameter/input mutation and nonfinite
fallbacks. Twelve CPU tests cover request/stage isolation, cache bounds,
mutation, hooks, frozen-backbone equality including nonpersistent buffers,
and cleanup after exceptions. Image conversion and PLY field checks match
their references exactly.

These CUDA comparisons are at operator level. Full-model comparisons below
use ROCm reference paths; no full-model NVIDIA CUDA comparison or upstream
dataset-level quality score is claimed.

## Full-pipeline agreement

Kidsroom mask 14, seed 42, default steps, both geometry decoders,
`compile=False`. Each device used one loaded model, alternating the earlier
optimized reference path and the final changes. The comparison retained shared
DINO weights and the expandable allocator on both paths. It toggled compact
CFG context, separate Q/K/V, dense RMS tail, small pose pointwise fusion,
image-feature reuse, FP16 linear storage and native GELU fusion. Controls retain
the preceding attention/normalization/layout optimizations.

After one warmup per path, three matched A/B pairs and three final-default
runs produced eight comparisons per device, including repeated controls.

| Maximum across the comparisons | R9700 | 8060S |
|---|---:|---:|
| Sparse occupancy IoU | 1.0 throughout | 1.0 throughout |
| Pose tensor absolute difference | 0 | 0 |
| SLAT feature RMS difference / reference RMS | 0.02499% | 0.02493% |
| SLAT maximum absolute difference | 0.02507 | 0.02128 |
| Gaussian-center symmetric NN p99 / bounding-box diagonal | 0.04921% | 0.05157% |
| Mesh symmetric NN p99 / bounding-box diagonal | 0.007245% | 0.007700% |

Preselected limits were exact occupancy/pose, feature normalized RMS <=1%, and
geometry p99 <=0.5% of the bounding-box diagonal. Geometry samples 10,000 points
in each direction against each complete point set, using a fixed random seed;
these are sampled nearest-neighbor distances, not a Hausdorff bound. Repeated
unchanged controls also exhibit small SLAT/mesh variation.

On R9700, a separate 27-mask paired sweep of the preceding combined changes
passed exact occupancy/pose, with maximum feature normalized RMS 0.05531%.
The subsequent 27-mask production GELU-fusion sweep also passed, with maxima
0.06034% feature normalized RMS, 0.02829 feature absolute difference,
0.06494% Gaussian-center p99 and 0.01586% mesh p99. The iGPU has the repeated
mask-14 comparisons above and complete default-path exports, rather than a
27-mask paired numerical sweep.

Both devices exported PLY/GLB for all 27 masks and assembled a scene of about
13.6 million Gaussians. Checks verify finite geometry and PLY attributes,
read back GLB meshes, and require the assembled count to equal the sum of
individual objects. Fresh-process output counts can vary, so agreement is
assessed within the paired runs rather than by equal cross-device counts.

## Performance and memory

Devices ran separately with no competing GPU workload. Timings synchronize the
GPU before and after inference, include preprocessing and both decoders, and
exclude loading, export and diagnostic serialization. CPU FP32 master weights
permit alternating storage formats outside the clock without retaining inactive
GPU copies. These masters are released before three final-default runs.

| Warm mask-14 inference | R9700 | 8060S |
|---|---:|---:|
| Earlier optimized path, median of three controls | 15.66 s | 63.14 s |
| New changes, median of three matched candidates | 12.97 s | 50.96 s |
| Final defaults, median of three runs | **12.99 s** | **50.59 s** |
| Final-default range | 12.99–13.27 s | 50.53–51.54 s |
| Final-default peak allocated GPU memory | 13.70 GiB | 13.70 GiB |
| Full 27-object run, peak allocated GPU memory | 14.74 GiB | 14.74 GiB |
| Full 27-object run, peak reserved GPU memory | 15.80 GiB | 16.19 GiB |
| Full-run CPU process peak RSS, including loading | 11.67 GiB | 11.64 GiB |

The first final-default run follows an allocator cache trim. Full-scene runs
retain individual Gaussian outputs until assembly. GPU allocation peaks above
exclude checkpoint loading. On Strix Halo GPU allocations use shared physical
RAM; do not add CPU RSS and GPU allocator statistics as an exact physical-memory
total.

DINO deduplication removes 3,652,472,832 bytes (3.40 GiB) of identical frozen
weights. Four stage-specific preprocessing/feature-selection wrappers remain;
byte comparison follows checkpoint loading, so the initial load-time peak is
not eliminated. Matching inputs produce four backbone computations and four
cross-stage cache hits per object. Frozen SS linear storage removes another
1,914,175,488 bytes (1.78 GiB) of FP32 weight storage. Normalization stays FP32.
A same-process allocator comparison preserved exact occupancy/pose while
reducing reserved memory from 21.34 GB to 15.06 GB on either device.

PLY attribute packing was separately measured on 13,611,264 Gaussians:
17.98 s to 0.062 ms, peak CPU process RSS 11.90 to 1.75 GiB, identical
925,565,952-byte attribute payload SHA-256. This isolates packing; it excludes
attribute preparation, device transfer and disk writing, and is not a full-run
RAM or export-time measurement.

The selected FP16/default-step configuration is the measured stopping point
for this tuning sweep. The final GELU fusion saved about 0.31–0.35 s in the
iGPU latent stage; tested HIP attention and persistent-graph alternatives did
not provide a worthwhile full-pipeline improvement at the required memory and
numerical behavior. This establishes diminishing returns among the tested
candidates, not an absolute hardware-optimality proof.

## Defaults and overrides

Set these variables inside the container before constructing the pipeline.
A value of `0` disables the respective optimization. Guards retain reference
paths for unsupported shapes, modes and devices.

| Optimization | Variable |
|---|---|
| Conditioning projections/casts | `SAM3D_CACHE_CROSS_ATTN` |
| Compact zero CFG context | `SAM3D_COMPACT_CFG` |
| SLAT attention | `SAM3D_TRITON_ATTN` |
| SLAT normalization / R9700 dense RMS tail | `SAM3D_TRITON_NORM` / `SAM3D_TRITON_DENSE_NORM` |
| Separate sparse Q/K/V | `SAM3D_SEPARATE_QKV` |
| Frozen linear FP16 storage / cast cache | `SAM3D_FP16_LINEAR_WEIGHTS` / `SAM3D_CACHE_LINEAR` |
| iGPU MLP layout / native GELU epilogue | `SAM3D_CACHE_LAYOUT` / `SAM3D_FUSED_GELU` |
| Frozen DINO sharing / raw feature reuse | `SAM3D_SHARE_BACKBONES` / `SAM3D_CACHE_IMAGE_FEATURES` |
| Exact elementwise fusion | `SAM3D_TRITON_POINTWISE` |

`PYTORCH_ALLOC_CONF=expandable_segments:True` is the image default; override
with `expandable_segments:False` for the allocator reference.
`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` enables fused SDPA on 8060S.
Optional rasterization, texture baking and layout optimization remain outside
the validated reconstruction/export paths, as described in the README.
