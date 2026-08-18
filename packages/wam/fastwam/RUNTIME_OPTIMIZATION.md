# Runtime Optimization Playbook - making diffusion / world-action models go real-time on AMD

General overview of *inference-time* optimizations for generative robot/world models, with the
FastWAM (Wan2.2 world-action MoT) speedups gathered so far as the first worked example. The goal
is to separate **general levers** (transfer to most diffusion-transformer inference) from
**model-specific levers** (depend on the architecture), record **what actually moved the needle
vs. what didn't** on our hardware, and give the next model a checklist toward real-time.

- **Hardware context:** AMD Strix Halo APU, `gfx1151` (RDNA3.5), ROCm 7.2.2, bf16. Empirical
  peaks (clean isolated): **37.8 TFLOP/s** compute, **230 GB/s** sustained memory BW (LPDDR5x-8000,
  256 GB/s theoretical), ridge **164 FLOP/byte**.
- **Detailed log / provenance:** `docs/RESEARCH_FINDINGS.md` (newest on top). Every number below
  is measured; raw metrics under `tmp/bench/`, plots under `artifacts/`.
- **Scope:** inference/deployment only. Nothing here trades training. Anything not bit-exact is
  flagged and kept **opt-in**.

---

## 0. TL;DR - decision table

| Lever | Kind | Targets | FastWAM result | Adopt? | Transfers to other models? |
|---|---|---|---|---|---|
| **Cache step-invariant compute** (context/text-embed + cross-attn K/V across denoise steps) | general | per-step redundant recompute | step 46→36 ms | **YES (default-on)** | **Yes** - any conditional diffusion sampler |
| **Cache episode-invariant compute** (prompt/text-encoder across replans) | general | per-plan redundant recompute | 852→699 ms (1.22×) | **YES (default-on)** | **Yes** - any model with a constant conditioning per episode |
| **Kill per-step host↔device copies / syncs** (device-resident RoPE freqs, buffers) | general | launch stalls + graph blockers | bit-exact, enables graph capture | **YES (micro-fix)** | **Yes** - universal hygiene |
| **Increase effective batch M** (CFG pair, best-of-N, parallel envs) | general | GPU under-utilization (tiny M) | full-step B=2 = +13% for **1.77×** throughput; B=16 = **5.6×** (knee at M=512) | *available* (throughput lever) | **Yes** - anything left of the roofline knee |
| **Fewer NFE / distillation** (fewer diffusion steps) | general | the ×N step loop directly | not yet tried | candidate | **Yes** - biggest single-plan lever if quality holds |
| **W8A8 int8 quant** (`torch._int_mm`) | general* | large GEMM weight bandwidth | ~1.07× (video FFN only); +1.4% action err | **NO** (quality cost, small win) | conditional - only for **big** GEMMs |
| **HIP graph capture** | conditional | CPU kernel-launch overhead | capturable & bit-exact but **0.999×** | **NO** (compute-bound here) | only if **launch-bound** |
| **torch.compile (Inductor)** | conditional | elementwise/norm tail fusion | 1.04× (not exact) | **NO** | marginal unless big elementwise tail |
| **Attention backend swap** (flash/efficient) | conditional | attention cost | default already ≈efficient; flash masked→unsupported | **N/A** | only if attention-bound |
| **hipBLASLt / QKV fusion / fp32 RoPE** | conditional | kernel selection / GEMM shape | 0.93-1.02× (neutral-to-worse) | **NO** | shape/stack dependent |

`*` general in principle, but on RDNA3.5 only int8×int8→int32 (`_int_mm`) is fast; fp8 `_scaled_mm`
is unsupported and weight-only+dequant is slower than bf16.

**FastWAM net so far:** **1.43-1.45× bit-exact** (852 → ~590 ms/plan) from caching alone, shipped
default-on. Everything else on this stack was neutral, not-exact, or a quality regression.

---

## 1. How to think about it - diagnose before you optimize

These models are **iterative denoisers**: a plan = (encode conditioning) + N × (denoise step) +
(decode). Two orthogonal questions decide which lever applies:

1. **Where is the time?** Split into: conditioning encode (once/plan), the ×N denoise loop
   (dominates), and decode (VAE - often skippable, see §3). *Profile first.* FastWAM: denoise loop
   52%, prefill 19%, text enc 17%, VAE enc 9%, **no VAE decode on the planning path**.

2. **What bounds each hot kernel - compute, memory bandwidth, or launch overhead?** Use a roofline:
  - **Arithmetic intensity** `AI = FLOPs / bytes`. Compare to the hardware **ridge** (peak
     FLOP/s ÷ peak GB/s). Above ridge → compute-bound; below → bandwidth-bound.
  - On gfx1151 the ridge is **≈150-214 FLOP/byte**. **Every** FastWAM GEMM sits at AI 30-115 ≪
     ridge → **100% bandwidth-bound**, running at only **2-43% of compute peak**.
  - Two structural causes for us: **tiny M** (32 action / 120 video rows sit far left of the
     M≈512 compute knee) and **narrow output dim** (N=1024 attn-out/ffn-down tile poorly).

The diagnosis dictates the lever:
- **bandwidth-bound & redundant across steps** → *cache it* (biggest, safest win - §2.1).
- **bandwidth-bound & tiny-M** → *grow M* (batching - §2.4) or *shrink weight bytes* (quant, big GEMMs only - §2.5).
- **launch-bound** → HIP graphs / compile (didn't apply to us - §4).
- **compute-bound** → fewer steps / distillation is the only real out (§2.6).

> Lesson from FastWAM: we confirmed the workload is bandwidth/tiny-GEMM bound, which is exactly
> why HIP graphs (launch overhead) and quant of the small action expert did nothing, while caching
> (removing whole redundant GEMMs) did everything. **Match the lever to the bottleneck.**

---

## 2. General optimizations (transfer across models)

### 2.1 Cache step-invariant computation across the denoise loop  ⭐ biggest safe win
In a conditional sampler the **conditioning is constant across all N steps**, yet naive code
recomputes it every step. Anything derived only from the conditioning can be computed once:
- the conditioning embedding / projection (`text_embedding(context)`),
- every cross-attention layer's **K and V** from the context (only Q changes per step).

FastWAM: caching the text cross-attn K/V + text-embed MLP once per plan cut the per-step cost
**46.0 → 35.6 ms**, contributing to the 1.43× total. **Bit-identical** (`max|Δ| = 0.0`) - it's
pure reuse of provably-constant tensors. **This is the #1 thing to check on any new diffusion
model** and is upstream-contributable (not a porting hack).

### 2.2 Cache episode-invariant computation across replans  ⭐
One level up: anything constant for the whole **episode** (not just one plan) should survive across
replans. FastWAM re-ran the 5.68 B UMT5-XXL text encoder (154 ms) on every replan for a fixed
instruction; memoizing `encode_prompt` by prompt string gave **852 → 699 ms (1.22×)**, bit-exact.
Generalizes to: fixed goal images, camera intrinsics, static scene tokens, etc.

### 2.3 Eliminate per-step host↔device copies and syncs
Small CPU→GPU copies inside the step (a) stall the pipeline every iteration and (b) make the step
**uncapturable** by HIP/CUDA graphs. FastWAM copied a **CPU** RoPE `freqs` buffer H2D *every step*;
moving it to the device once was **bit-exact**, removed 10 syncs/plan, and made the whole step
capturable. Universal hygiene - cheap, zero-risk, and a prerequisite for §4 if you ever need it.

### 2.4 Increase effective batch M (climb the roofline)  ⭐ untapped throughput lever
Tiny-M GEMMs waste the GPU. Because per-GEMM latency is **weight-read (bandwidth) dominated**,
extra rows are nearly free until you hit the compute knee (~M=512). Confirmed on the **real action
step** (`_predict_action_noise`, B=1..32), not just isolated GEMMs:

| B (M=32·B) | step ms | ms/sample | throughput vs B=1 |
|---|---|---|---|
| 1 (M=32) | 34.45 | 34.45 | 1.00× |
| 2 (M=64) | 38.92 (+13%) | 19.46 | **1.77×** |
| 4 (M=128) | 55.59 | 13.90 | 2.48× |
| 8 (M=256) | 73.76 | 9.22 | 3.74× |
| 16 (M=512) | 98.38 | 6.15 | **5.60×** |
| 32 (M=1024) | 196.28 | 6.13 | 5.62× (saturated) |

The **knee lands exactly at M=512 (B=16)** as the roofline predicted; **B=32 saturates**
(per-sample flat, now compute-bound). Doubling the whole step costs only +13% (the isolated FFN GEMM
alone was +0.6%; the full step also grows attention QKV/O and elementwise with M). Sweet spot B=8-16.
Ways to spend the near-free rows (quality-neutral or quality-*improving*):
- **CFG**: batch cond+uncond into one M=64 pass - guidance for ~+1% instead of 2×.
- **Best-of-N sampling**: N candidate actions in one batch → better plans for ~free at small N.
- **Parallel-env eval**: batch environments for ~2-4× wall-clock throughput.

> ⚠️ **This is a throughput lever, NOT a real-time latency lever - read before applying.**
> Batching does **not** make a single sequential plan faster. It only helps when you have
> *independent* work to run in the same instant. Two hard sequential dependencies block any
> "free M" for a lone plan: (1) the **denoise loop is sequential** - step *t* consumes step
> *t−1*'s output, so the 10 steps can't be batched (latency = 10 × one step, period); (2) the
> **rollout is sequential** - plan *k+1* depends on the actions executed from plan *k*, so there's
> no future plan to batch with. The action horizon's 32 tokens are *already* the M=32 inside one
> pass; you can't inflate them for one sample.
>
> Where the free rows actually come from, and whether we have them:
> - **Parallel-environment / multi-episode eval (our benchmark suites) - REAL for us.** Batching N
>   episodes' plans → ~5.6× at B=16, so RoboTwin/LIBERO suites finish ~5× faster. Speeds up *our*
>   iteration, not the robot.
> - **CFG (guidance) - only if we add it.** cond+uncond are two independent passes at the *same*
>   step → batch to +13% instead of +100%. Makes a quality upgrade nearly free; not a speedup of
>   today's conditional-only path.
> - **Best-of-N sampling - only if we add it.** N candidate trajectories batch together → better
>   plans for sublinear cost. Improves quality-per-second, not single-sample latency.
> - **Serving a fleet (many robots, one GPU) - not our case today.** Concurrent requests batch →
>   more robots per GPU; a cost/throughput win, not a per-robot latency win.
>
> **Bottom line:** for real-time control of one robot, batching buys nothing - latency there comes
> from caching (§2.1-2.2), fewer NFE (§2.6), and faster kernels. Batching's payoff is the eval
> harness and any future multi-sample / multi-robot serving.

### 2.5 Quantization - only for the *large* GEMMs (conditional)
On RDNA3.5 the only fast path is **W8A8 dynamic** (per-token act int8 + per-channel weight int8 +
`torch._int_mm` int8×int8→int32 + dequant). fp8 `_scaled_mm` is unsupported; weight-only+dequant is
slower than bf16. Break-even is `max(in,out) ≳ 8k` - below that the quant/dequant overhead *slows*
the GEMM. FastWAM: only the video expert FFN (ffn=14336) benefits, but it runs **once** while the
small action expert (ffn=4096, can't benefit) runs **10×** → e2e ceiling **~1.07×** with a
measurable closed-loop regression (one precision-sensitive task 10/10 → 9/10). **Built as opt-in,
default-OFF, not adopted.** Rule of thumb: quant pays off when your hot loop is *big-GEMM* bound;
it does nothing for tiny-GEMM/overhead-bound loops.

### 2.6 Fewer denoise steps (NFE) / distillation - the compute-bound escape hatch
The step loop runs N× by definition, so **halving N ≈ halves the loop**. This is the lever when a
model is genuinely compute-bound (where caching/batching/quant can't help). Not yet tried on
FastWAM (10 steps); a distillation / consistency / fewer-step schedule is the highest-leverage
*single-plan-latency* direction remaining and is broadly applicable to all diffusion models. Cost
is potential quality loss → must be closed-loop validated.

---

## 3. Model-specific optimizations (architecture-dependent)

### 3.1 Exploit dual-expert / prefill structure (FastWAM MoT)
Wan2.2 world-action is a Mixture-of-Transformers: a big **video expert** and a small **action
expert**. The planning path (`infer_action`) **prefills the video branch once** into per-layer
K/V, then each denoise step runs *only the small action expert* against the cached video K/V. This
architectural split is why the loop is cheap (35 ms, not 300 ms) and why our caching lands there.
Transfers only to models with a similar big-context / small-query-loop split.

### 3.2 Skip the decoder when the task doesn't need pixels
FastWAM's full-video path (`infer_joint`, ~5.5 s) spends **2.2 s (40%) in VAE decode**. The
deployed policy needs *actions*, not frames, so the planning path **never decodes** → ~6.2× cheaper
than full video before any other optimization. Generalizes: if a downstream consumer needs latents
or actions rather than RGB, don't run the decoder. Often the single biggest structural win.

### 3.3 Right-size the sampling path to the deployment
Deployed dims (action_dim 14, horizon 32, replan every 24, N=10 steps, 1 frame of video context)
keep the tensors tiny - which is *good* for latency but *bad* for GPU utilization (see §2.4). The
model-specific knob is choosing horizon / replan cadence / step count that meets the control loop's
real-time budget, not the generation-quality budget.

---

## 4. Things that did NOT help here (record so we don't retry blindly)

All measured on FastWAM/gfx1151; baseline = caching default ~590 ms. These are **stack/bottleneck
specific negatives**, not universal - re-evaluate on different hardware or a launch-bound model.

| Lever | Result | Why it didn't help *here* |
|---|---|---|
| **HIP graph capture** | 0.999× (bit-exact) | Workload is **compute/bandwidth-bound, not launch-bound** - removing CPU launch overhead buys nothing. (Still a valid lever for launch-bound models.) |
| **torch.compile** | 1.04× (not exact) | Inductor only fused the ~15% elementwise/norm tail; the 72% GEMM floor is untouched; +27 s compile. |
| **QKV fusion (3→1 GEMM)** | 0.93× (**slower**) | Bigger-N GEMM tiles *worse* at M=32; splitting was faster. Counterintuitive at tiny M. |
| **hipBLASLt on** | 0.99× | No better small-GEMM kernels for these shapes on this ROCm. |
| **RoPE fp64→fp32** | 1.02× (not exact) | Marginal; not bit-exact so not worth it. |
| **`inference_mode`** | 1.00× | Negligible. |
| **Attention backend swap** | ~neutral | torch default already dispatches to memory-efficient (~4× faster than math); AOTriton flash is *unsupported with an attn_mask* on gfx1151 and only marginally faster maskless. Attention is ~1.5 ms of a 35 ms step anyway. |

> Meta-lesson: at **tiny M**, "obvious" GEMM tricks (fusion, better BLAS) can backfire, and
> launch-overhead tools (graphs/compile) are no-ops because you're not launch-bound. Profile the
> bottleneck class *before* spending time here.

---

## 5. Hardware baseline - gfx1151 (Strix Halo), bf16

- Compute peak **37.8 TFLOP/s**; sustained memory BW **230 GB/s** (256 theoretical, LPDDR5x-8000;
  clean isolated re-measure - the earlier 244 was a best-case single-GEMM read, not sustained triad).
- **Roofline ridge ≈ 164 FLOP/byte.** Below → bandwidth-bound (most inference GEMMs).
- **Compute knee at M ≈ 512** rows for ≥80% peak; M=32 ≈ 15-20%, M=120 ≈ 40%.
- "Best" GEMM shapes: **M ≥ 512, wide balanced dims (N ≥ 4096)**. Thin/small-M GEMMs (FastWAM's
  regime) waste both compute and bandwidth (N=1024 outputs hit only 2-17% of BW).
- Notes: fp8 `_scaled_mm` unsupported; `_int_mm` int8 is fast; deep HIP-graph replay queues (200+
  without sync) hang on ROCm 7.2.2 - sync per plan-worth.

---

## 6. A checklist for the next model → real-time

1. **Profile** the plan into encode / step-loop / decode; get per-kernel bottleneck class (roofline).
2. **Skip the decoder** if the consumer doesn't need pixels (§3.2) - often the biggest win.
3. **Cache** everything invariant: across steps (§2.1) and across episodes/replans (§2.2). Verify bit-exact.
4. **Remove per-step H2D copies/syncs** (§2.3) - free hygiene.
5. If under-utilized (tiny M) and you can spend rows: **batch M** for CFG / best-of-N / parallel envs (§2.4).
6. If a **big-GEMM** loop dominates: **W8A8 int8** (§2.5), closed-loop validated.
7. If **compute-bound**: **reduce NFE / distill** (§2.6) - the only real single-plan lever left.
8. Only if profiling shows **launch-bound**: HIP graphs / torch.compile (§4).
9. **Always** re-run closed-loop task success for anything non-bit-exact before adopting.

---

## 7. Per-model ledger

### FastWAM (Wan2.2 world-action MoT, 12.4 B) - planning path `infer_action`
- **Net shipped: 1.43-1.45× bit-exact** (852 → ~590 ms/plan), default-on, kill-switch
  `FASTWAM_TEXT_KV_CACHE=0`. In-source patch `packages/wam/fastwam/patches/fastwam_kv_cache.patch`,
  baked into `fastwam-robotwin` / `fastwam-libero` / `fastwam` images.
- Breakdown before opt (10 steps): denoise loop 468 ms (52%), video prefill 173 (19%), text enc
  154 (17%), VAE enc 83 (9%), no decode. Full-video path `infer_joint` ~5.5 s (VAE decode 2.2 s).
- Wins: text-encode episode cache (1.22×) + cross-attn K/V step cache (→1.43×), device-resident
  RoPE freqs (bit-exact micro-fix).
- Evaluated, not adopted: HIP graph (0.999×), torch.compile (1.04×, inexact), W8A8 quant (1.07×,
  +1.4% action err, 9/10 on one task).
- Open levers: M-batching for CFG/best-of-N/parallel-env (§2.4, full-step confirmed: +13% for 1.77×
  at B=2, up to 5.6× at B=16, knee M=512), fewer NFE.

### (template for the next model)
- Net shipped: … / bottleneck class: … / structural wins (decoder skip, caches): … / evaluated-not-adopted: … / open levers: …
