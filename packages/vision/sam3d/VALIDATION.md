# Numerical and performance validation

Validated 2026-10-20 on Radeon 8060S (`gfx1151`) and Radeon AI PRO R9700
(`gfx1201`), ROCm 7.14 / PyTorch 2.12. SAM 3D source is pinned in the Dockerfile;
weights are `facebook/sam-3d-objects` revision
`2e73555018d2741ccd486e56c24fac41155a1dc6`.

## Independent numerical checks

`check_sam3d_numerics.py --golden-root ...` passed on both devices in the rebuilt
image. It verifies SHA-256 hashes before loading the trusted reference fixtures.
The pinned spconv-triton checkout includes generators that require unchanged
CUDA spconv; no NVIDIA GPU was available for a new full-model CUDA run.

| Check | R9700 maximum absolute error | 8060S maximum absolute error |
|---|---:|---:|
| 32 float32 sparse forward cases | 8.58e-6 | 8.58e-6 |
| 7 float16 sparse forward cases | 9.76e-4 | 9.79e-4 |
| float32 attention vs CPU float64 | 4.99e-7 | 4.99e-7 |
| float16 attention vs CPU float64 | 6.28e-4 | 6.28e-4 |
| bfloat16 attention vs CPU float64 | 4.42e-3 | 4.42e-3 |

Sparse output coordinates and shapes match the CUDA fixtures exactly. Value
comparisons use the fixtures' existing absolute tolerances, with relative
tolerance zero. The older loose float16 cases are excluded; the tight family
bounds error against CUDA float32 results using CUDA's own float16 rounding.
Cases cover channel tails, up to 256 channels, multiple/empty batches,
submanifold, strided, dilated, transposed and inverse convolution.

Attention checks include self-attention, unequal query/key lengths, independent
blocks and a zero-length query block. Predetermined absolute tolerances are
2e-5 / 2e-3 / 2e-2 for float32 / float16 / bfloat16 on these unit-scale inputs.
These are test-input bounds, not universal error guarantees for arbitrary values.

A real checkpoint's 128-channel 3x3x3 sparse convolution was also checked at
32 output positions against a CPU float64 gather/matmul reference, after applying
the same float16 input/weight rounding. RMS error divided by reference RMS was
0.0205% on R9700 and 0.0230% on 8060S; both passed the preselected mixed
absolute/relative tolerance (2e-3 each).

For 32 rows of real 5,252-token attention, optimized maximum absolute error was
0.001905 / 0.001723 on R9700 / 8060S. The unchanged iGPU baseline exceeded the
2e-3 absolute threshold in one of 32,768 values (error 0.002019). The optimized
sample passed the original threshold. An additional strict two-output-ULP
diagnostic fails for both implementations around cancellation near zero; neither
implementation is correctly rounded element by element against float64. The
raw diagnostic failures are retained in the local reports.

## Full-pipeline agreement: attention change

Kidsroom mask 14, seed 42, upstream default inference steps, both decoders,
`compile=False`. The reference is the previous ROCm package, before replacing
block-diagonal masks with independent unmasked SDPA calls.

| Optimized vs baseline | R9700 | 8060S |
|---|---:|---:|
| Sparse occupancy IoU | 1.0 | 1.0 |
| Pose tensors | exact | exact |
| SLAT feature RMS difference / baseline RMS | 0.0241% | 0.0242% |
| Repeating unchanged baseline: feature RMS difference | 0.0245% | 0.0250% |
| Mesh symmetric nearest-neighbor p99 / bounding-box diagonal | 0.00622% | 0.00592% |
| Gaussian-center symmetric nearest-neighbor p99 / diagonal | 0.0475% | 0.0479% |

Geometry distances sample 10,000 points in each direction against each complete
point set, using a fixed random seed. They are sampled distances, not a global
Hausdorff bound. Before the A/B runs, acceptance limits were fixed at unchanged
occupancy, feature RMS difference <=1%, and geometry p99 <=0.5% of the diagonal.
Both devices passed; changes are comparable to baseline repeat variation.

Between AMD devices, optimized occupancy IoU is 99.636%. Mesh nearest-neighbor
median / p99 distance is 0.068% / 0.425% of the diagonal; the sampled maximum is
2.176%. These outputs agree closely but are not bitwise identical. This does not
establish full-model parity with CUDA or validate arbitrary images.

## Export regression checks: attention change

The rebuilt image passed all 27 upstream kidsroom masks on R9700 and assembled
a 13,611,264-Gaussian scene. On 8060S, masks 14 and 3 passed and assembled a
1,401,472-Gaussian scene. PLY/GLB exports were reloaded and checked for finite,
nonempty geometry; PLY attributes were checked after decoding opacity logits.
These checks cover both inference stages and both decoders. Fresh-process
output counts are representative and can vary slightly even with seed 42.
The canonical `ryzers build --name sam3d sam3d` also passed.

## Performance: attention change

| GPU | Baseline | Optimized | Speedup | Decoder baseline -> optimized |
|---|---:|---:|---:|---:|
| Radeon 8060S | 127.3 s | 86.5 s | 1.47x | 12.49 -> 1.31 s |
| Radeon AI PRO R9700 | 25.1 s | 20.7 s | 1.21x | 2.39 -> 0.57 s |

Each implementation ran three times in one process; its first run was discarded
and the table uses the median of the remaining two. Model loading, export and
post-timing tensor capture are excluded. GPU synchronization brackets stages;
GPU events measure sparse attention. These are workstation measurements with
background CPU activity, not isolated maximum-throughput benchmarks. Both
implementations retain the same precision, checkpoint and inference steps.
Peak PyTorch allocation remains approximately 20.3 GB; another stage sets the peak.

The optimization avoids a quadratic mask, uses fused attention for a single
sequence, and computes independent spatial windows separately. R9700 profiling
still attributes about 9.76 s of device work to fused attention, 3.59 s to linear
`addmm` operations and 1.71 s to copies. These operator categories must not be
added to nested profiler annotations or duplicate kernel entries.

Remaining candidates require measurement and numerical regression checks:

- Compile/fuse the repeated transformer pointwise operations and reduce small
  device copies and launches. The profile recorded about 178,000 kernel launches;
  CPU time in blocking HIP copies includes waiting and is not a transfer-only cost.
- Tune attention/GEMM kernels for these architectures. Sparse-structure generation
  still costs about 9.3 s / 47.1 s on R9700 / 8060S after this change.
- Reuse the raw depth-model prediction for multiple objects in the same image.
  Keep intrinsics and reapply mask-dependent clipping separately; caching the
  completed point map would change behavior. Warm depth inference is only about
  0.4 s / 0.8 s per object, so its benefit is smaller than the attention change.

No claim is made that this implementation is bug-free or maximally performant.

## Cross-attention conditioning cache

The next optimization reuses image-conditioning casts, CFG zero tensors and
cross-attention K/V linear projections within each sampling stage. It leaves
latent self-attention, model precision and inference steps unchanged. Conditional
and unconditional inputs are distinct cache keys. Context-local storage holds
input references to prevent identity reuse; tensor and parameter mutation
counters and autocast settings invalidate entries. Projection caching bypasses
training modules; every cache bypasses gradient-enabled calls and tensors without
mutation counters. Each stage
retains at most 256 entries / 3 GiB of cached outputs and clears them on return
or exception, including between masks. Input references also remain alive until
the stage ends. `SAM3D_CACHE_CROSS_ATTN=0` disables reuse.

Six CPU tests cover CFG/request separation, input/output/weight mutation,
parameter replacement, autocast/dtype changes, gradient/training fallback,
inference-mode fallback and exception cleanup. Run them with
`python /ryzers/test_sam3d_conditioning_cache.py` in the image.

The cache experiment compares cached and uncached paths in the same build
(`SAM3D_CACHE_CROSS_ATTN=1` / `0`). The uncached path retains commit `85289e7`'s
independent unmasked SDPA behavior, with the added Python cache guards. Baseline
and cached runs alternate in one process after a discarded warmup, with identical checkpoint, mask 14,
seed 42, default steps and both decoders. GPU synchronization brackets stages.
Wall times include small CPU captures of intermediate sparse tensors; final
geometry capture and comparison happen after timing. Background CPU workloads
remain present, so these are workstation latency measurements.

| Additional cache optimization | Uncached median | Cached median | Speedup | Sparse structure | Sparse latent |
|---|---:|---:|---:|---:|---:|
| R9700 | 21.31 s | 20.30 s | 1.050x | 9.34 -> 8.61 s | 9.25 -> 9.03 s |
| 8060S | 88.14 s | 79.75 s | 1.105x | 50.27 -> 43.86 s | 34.66 -> 32.69 s |

Three warm runs per mode. R9700 controls ranged 21.27–22.94 s and cached runs
20.29–20.33 s; 8060S controls ranged 85.55–88.77 s and cached runs 79.34–80.00 s.
Every cached run beat every control on its device. The cache avoided 2,880 K/V
projections plus 108 casts/zeroing operations per inference. Maximum cached
output storage was 3.04 GB (2.83 GiB) in one stage. Mask-14 peak PyTorch allocation
remained 20.3 GB on both devices; this is allocated tensor memory, not total
driver/allocator-reserved memory. A 1 or 2 GiB output limit caused repeat misses;
3 GiB fits this conditioning workload. Larger workloads fall back to recomputing
entries that do not fit the bound.

All three cached runs were compared with the first warm uncached run on each
device using the previously fixed acceptance limits above. Occupancy and pose
were exact. The following are the worst observed differences across those runs:

| Cached vs uncached | R9700 | 8060S |
|---|---:|---:|
| SLAT normalized RMS difference | 0.0257% | 0.0330% |
| SLAT maximum absolute difference | 0.01776 | 0.04175 |
| Uncached repeat: normalized RMS difference | 0.0266% | 0.0437% |
| Uncached repeat: maximum absolute difference | 0.01679 | 0.07269 |
| Mesh NN p99 / diagonal | 0.00810% | 0.00570% |
| Mesh sampled maximum / diagonal | 0.1867% | 0.2655% |
| Gaussian-center NN p99 / diagonal | 0.04757% | 0.04622% |
| Gaussian-center sampled maximum / diagonal | 0.2718% | 0.2709% |

Geometry uses the same 10,000-point symmetric sampling as above; sampled maxima
are not global error bounds. Full-model outputs are not bitwise identical, and
these tests characterize one mask's numerical agreement. Between devices, the
final cached runs had occupancy IoU 99.678%, mesh NN median / p99 0.0654% /
0.4790% of the diagonal, and sampled maximum 2.065%.

The rebuilt final image also passed the independent CPU float64 attention checks
and all 39 frozen CUDA sparse forward cases on each GPU, with the same maximum
errors in the first table. This remains operator-level CUDA validation; there
was no full-model NVIDIA CUDA reference run.

Final cache-enabled export regression passed all 27 kidsroom masks on R9700,
including both decoders, PLY/GLB reload/finite checks, and assembly of a
13,611,264-Gaussian scene. The 8060S passed masks 14 and 3 with the same checks,
assembling 1,401,408 Gaussians. These are functional export checks; the numerical
A/B comparison above covers mask 14. Both regression processes exited cleanly.

### Other lessons evaluated from vllm-radiance

Source inspected at `1d2e24849c9d8441b9614acade5252c9937a2bbf`:
[offline TunableOp workflow](https://github.com/magiccodingman/vllm-radiance/blob/1d2e24849c9d8441b9614acade5252c9937a2bbf/docs/FP8_KV_TUNABLEOP.md),
[vision attention](https://github.com/magiccodingman/vllm-radiance/blob/1d2e24849c9d8441b9614acade5252c9937a2bbf/radiance_vit_attn.py).

Native packed variable-length attention passed sampled float64 checks on both
GPUs and sped up a decoder-shaped operator probe, but full-pipeline timings did
not establish a reliable additional gain. Offline per-architecture GEMM tuning
also passed sampled numerical checks, but selected replay failed to improve
the R9700 whole pipeline and iGPU baseline drift exceeded the apparent gain.
Neither experiment changes the defaults. Their LLM-specific paged/causal
attention and low-precision formats are not direct replacements for SAM3D's
noncausal FP16 attention; no quantization is introduced here.
