# Numerical and performance validation

Validated 2026-10-20 on Radeon 8060S (`gfx1151`) and Radeon AI PRO R9700
(`gfx1201`), ROCm 7.14 / PyTorch 2.12. SAM 3D source is pinned in the Dockerfile;
weights are `facebook/sam-3d-objects` revision
`2e73555018d2741ccd486e56c24fac41155a1dc6`.

## Reproduce the checks

`ryzers run` runs only the installation smoke test, without weights or credentials.
Full-model validation is explicit and requires model access as described in the
README. These commands run the numerical checks and cache lifecycle tests:

```sh
ryzers run "python /ryzers/check_sam3d_numerics.py"
ryzers run "python /ryzers/test_sam3d_conditioning_cache.py"
```

For the frozen CUDA fixtures, use the same trusted checkout as the Dockerfile:

```sh
ryzers run bash
# Inside the container:
git clone https://github.com/L-Reichardt/spconv-triton.git /tmp/spconv-reference
git -C /tmp/spconv-reference checkout 63361695e891de7aff32841ebb86f2661a41b059
python /ryzers/check_sam3d_numerics.py --golden-root /tmp/spconv-reference
```

The checker verifies fixture hashes. Reports persist at
`workspace/sam3d/numerics.json`. For exported geometry and scene assembly:

```sh
ryzers run "python /ryzers/test_sam3d.py --mask-index 14 3"
ryzers run "python /ryzers/test_sam3d.py --all-masks"
```

These explicit checks can take minutes and download large model checkpoints;
they are not part of the default installation test.

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


## Architecture-tuned sparse-latent attention

Validated 2026-10-21. A streaming-softmax Triton kernel handles single-block
FP16 attention with 16 heads, 64 channels per head and at least 2,048 queries
and keys, only during `sample_slat` on gfx1151/gfx1201. It accumulates scores,
softmax statistics and output in float32, rounding softmax probabilities to
float16 before the value product. It writes contiguous token/head/channel
output directly. Runtime token counts let different masks share compiled code.
Structure sampling, other geometries/dtypes/devices and gradient-enabled calls
retain PyTorch SDPA. `SAM3D_TRITON_ATTN=0` selects that reference path.

The checker adds two packed-stride/tail cases (2,049/2,071 and 4,096/4,096
queries/keys), sampling 32 queries against every key/value in CPU float64.
Both GPUs selected Triton and passed the predetermined absolute tolerance
2e-3, relative tolerance zero. Maximum absolute errors were 6.24e-5 on R9700
and 1.07e-4 on 8060S. The previous nine attention checks and all 39 frozen CUDA
sparse forward cases also passed with unchanged maximum errors. Six cache
lifecycle tests passed, including stage cleanup when conditioning reuse is off.

An additional real-checkpoint capture uses 5,252 queries with 5,252/5,496
keys for self/cross attention. The same R9700-captured FP16 operands were
replayed on both architectures, sampling 32 query rows against complete keys.
Maximum absolute errors against CPU float64 were 0.001043 / 0.001054 on
R9700 / 8060S, passing 2e-3 with relative tolerance zero. Maximum normalized
RMS error was 0.0211%. These are sampled operator checks; they do not establish
universal error bounds or full-model CUDA agreement.

The packaged A/B experiment keeps conditioning reuse enabled throughout and
alternates `SAM3D_TRITON_ATTN=0` / `1` in one process: mask 14, seed 42,
default steps, both decoders and `compile=False`. Discard the first run of each
path; report medians of three subsequent warm runs per path. Model loading,
export and final geometry capture are excluded; small intermediate CPU tensor
captures are included, with synchronization around each stage.

| GPU | PyTorch SDPA | Triton SLAT attention | Speedup | SLAT stage |
|---|---:|---:|---:|---:|
| R9700 | 20.92 s | 17.53 s | 1.194x | 9.03 -> 6.27 s |
| 8060S | 79.42 s | 67.89 s | 1.170x | 32.59 -> 21.31 s |

Every candidate beat every control on its device. Peak PyTorch allocation
remained 20.3 GB. Each of the three candidate runs passed the existing gates
against the first warm control: exact occupancy and pose, feature NRMSE <=1%
and sampled symmetric geometry NN p99 <=0.5% of the bounding-box diagonal.
Worst observed differences across the three runs:

| Triton vs PyTorch | R9700 | 8060S |
|---|---:|---:|
| SLAT normalized RMS difference | 0.02593% | 0.02579% |
| SLAT maximum absolute difference | 0.02629 | 0.03242 |
| Unchanged control repeat NRMSE | 0.02634% | 0.03554% |
| Unchanged control repeat maximum absolute difference | 0.02242 | 0.05169 |
| Mesh NN p99 / diagonal | 0.00625% | 0.00793% |
| Mesh sampled maximum / diagonal | 0.20385% | 0.16230% |
| Gaussian-center NN p99 / diagonal | 0.04871% | 0.04713% |
| Gaussian-center sampled maximum / diagonal | 0.25351% | 0.22252% |

Geometry uses the same fixed 10,000-point symmetric sampling described above.
These timings are workstation measurements. They demonstrate a further gain,
not a hardware latency floor; baseline repeat variation and sampled error
limits still apply. Broader export checks for preceding optimizations are
reported separately above.

## Sparse-latent normalization and linear-parameter reuse

The subsequent optimization fuses FP32 sparse RMS normalization and LayerNorm32
into Triton kernels during `sample_slat` only. Structure normalization and all
decoder normalization retain upstream PyTorch. Guards restrict RMS inputs to
FP16 with 16 heads / 64 channels, and layer normalization to contiguous FP16
matrices with 1,024/2,048 channels. Training, gradients, other devices, shapes
and dtypes use the original computation. `SAM3D_TRITON_NORM=0` disables fusion.

A normal `nn.Linear` subclass additionally reuses FP32-to-FP16 parameter casts
inside sparse-structure sampling, under FP16 CUDA autocast. It preserves
parameter/state-dict names and applies to the SAM transformer's dense attention
and MLP modules. A separate context-local cache holds at most 2,048 entries /
2 GiB of cast outputs, independently of the 3 GiB conditioning-output cache.
Input/output mutation counters invalidate entries, strong references prevent
identity reuse, and return/exception clears the stage. Training, gradients,
inference tensors without mutation counters and other precisions/devices fall
back to PyTorch. `SAM3D_CACHE_LINEAR=0` disables this parameter reuse.

Eight CPU cache tests passed, including parameter-cache nesting, mutation,
output mutation, disable flags, capacity fallback and exception cleanup. GPU
parameter checks compare four calls, including weight mutation and bias
replacement, against uncached FP16 autocast on frozen parameters. Both devices
matched exactly: maximum absolute difference zero, four cache hits/four misses.

The numerical checker adds six normalization cases: packed/strided sparse RMS
with autocast on/off, and 1,024/2,048-channel layer normalization on zero-centered
and offset inputs. Against independent CPU float64 formulas, maximum absolute
error was 0.001953 on R9700 and 0.001950 on 8060S. All passed preselected mixed
absolute/relative tolerances of 2e-3 each. The remaining attention and 39 CUDA
sparse fixture checks also passed on both GPUs.

Real checkpoint inputs were independently checked at 32 rows per shape. Sparse
RMS maximum absolute error was 0.003441 / 0.003852 on R9700 / 8060S; SLAT layer
normalization was 0.003829 / 0.003614. A separate 2,048-channel decoder-shaped
operator replay reached 0.02963 / 0.02992 on larger-magnitude outputs. That
replay checks the kernel; decoder execution still retains PyTorch. Every sampled
value passed the same mixed 2e-3 / 2e-3 bound; maximum normalized RMS error
across the three shapes was 0.0242% / 0.0261%. These are mixed tolerances,
not a universal 2e-3 absolute bound.

A/B uses the same packaged image, with Triton SLAT attention and conditioning
reuse enabled in both paths. Alternate normalization/parameter caching off and
on, discard one warmup per path, then compare three warm runs each: mask 14,
seed 42, default steps, both decoders, `compile=False`, the same synchronized
stage timing and capture method as the preceding experiment.

| GPU | Additional optimizations off | On | Speedup | Structure stage | SLAT stage |
|---|---:|---:|---:|---:|---:|
| R9700 | 18.09 s | 16.23 s | 1.114x | 8.64 -> 8.18 s | 6.26 -> 5.50 s |
| 8060S | 68.39 s | 65.01 s | 1.052x | 43.71 -> 43.04 s | 21.45 -> 18.78 s |

R9700 controls ranged 17.36–18.26 s and candidates 16.08–16.78 s; 8060S controls
67.87–68.51 s and candidates 64.93–65.31 s. Every candidate beat every control
on its device. Peak PyTorch allocation remained 20.3 GB. This is a separate
experiment from the attention table; speedup ratios are not multiplied.

All three candidates per GPU passed the existing acceptance gates against the
first warm control, including exact occupancy and pose. Worst observed values:

| Final additional optimizations vs control | R9700 | 8060S |
|---|---:|---:|
| SLAT normalized RMS difference | 0.02432% | 0.02550% |
| SLAT maximum absolute difference | 0.02034 | 0.02980 |
| Control repeat NRMSE | 0.02410% | 0.02480% |
| Control repeat maximum absolute difference | 0.01953 | 0.01894 |
| Mesh NN p99 / diagonal | 0.00625% | 0.00544% |
| Mesh sampled maximum / diagonal | 0.18442% | 0.17573% |
| Gaussian-center NN p99 / diagonal | 0.04507% | 0.05236% |
| Gaussian-center sampled maximum / diagonal | 0.23667% | 0.31727% |

The fixed limits and 10,000-point symmetric geometry sampling remain unchanged.
The rebuilt package retains a quick default smoke: a 1x1 sparse operation on
GPU against dense CPU PyTorch using CUDA spconv's 1x1 storage convention.
Explicit reconstruction checks retain the 3x3x3 sparse/dense comparison; the
full numerical suite retains its 39 frozen CUDA cases.

### Further candidates and the performance floor

These measurements establish additional validated gains, not a hardware floor.
A wider normalization rewrite and faster structure-attention variants were
rejected because they changed occupancy/pose on 8060S. A repeated structure
HIP-graph experiment was slower including request-local warmup/capture: R9700
structure sampling grew from 8.61 s to 9.88–10.04 s. Basic sparse-metadata reuse
changed the SLAT stage by less than 0.2%, without establishing a reliable gain.
MLP compilation and alternative weight layouts also failed to establish a
consistent gain on both devices. None of those variants is enabled.

Attention tuning evaluated tile sizes, warps, pipeline stages and occupancy
settings on both architectures. Additional reciprocal/reduction configurations
still differed from PyTorch on real structure-attention inputs; no faster
structure kernel is promoted. Reduced precision, fewer denoising steps and
relaxed occupancy/pose gates are not part of these results. Further speedups
need new measured evidence and numerical qualification.
