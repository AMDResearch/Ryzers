# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Planning-latency benchmark + per-part breakdown for FastWAM on Strix Halo (gfx1151).

Times a warm infer_action (num_inference_steps=20) split into text encoder, vision encoder,
world prefill and diffusion plan, and compares SDPA backends (MATH vs FLASH/AOTriton vs
default) to show the flash-attention effect.

Env: FASTWAM_REPO, CKPT, CONFIG_NAME, NUM_STEPS(20), BENCH_ITERS(5).
"""
import os
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

FASTWAM_REPO = os.environ.get("FASTWAM_REPO", "/repos/fastwam")
CKPT = os.environ.get("CKPT", "/models/fastwam_release/libero_uncond_2cam224.pt")
CONFIG_NAME = os.environ.get("CONFIG_NAME", "sim_libero")
NUM_STEPS = int(os.environ.get("NUM_STEPS") or "20")
ITERS = int(os.environ.get("BENCH_ITERS") or "5")
PROMPT = os.environ.get("PROMPT", "pick up the object and place it")
_DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}
DTYPE = _DTYPES[os.environ.get("DTYPE", "bf16").lower()]


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(fn, n=ITERS, warmup=2):
    for _ in range(warmup):
        fn()
    _sync()
    t = time.time()
    for _ in range(n):
        fn()
    _sync()
    return (time.time() - t) / n * 1000.0


def _compose_cfg():
    from omegaconf import OmegaConf
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    for name, fn in (("eval", eval), ("max", lambda x: max(x)),
                     ("split", lambda s, idx: s.split("/")[int(idx)])):
        try:
            OmegaConf.register_new_resolver(name, fn, replace=True)
        except Exception:
            pass
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.path.join(FASTWAM_REPO, "configs"), version_base="1.3"):
        return compose(config_name=CONFIG_NAME, overrides=[f"ckpt={CKPT}"])


def main() -> int:
    print(f"torch {torch.__version__}  hip={torch.version.hip}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}")
    for n in ("flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled"):
        fn = getattr(torch.backends.cuda, n, None)
        print(f"  {n}: {fn() if fn else 'n/a'}")

    if FASTWAM_REPO not in sys.path:
        sys.path.insert(0, FASTWAM_REPO)
    from hydra.utils import instantiate
    cfg = _compose_cfg()
    height, width = (int(x) for x in cfg.data.train.video_size)
    action_horizon = int(cfg.data.train.num_frames) - 1
    proprio_dim = int(cfg.data.train.processor.proprio_output_dim)

    t0 = time.time()
    model = instantiate(cfg.model, model_dtype=DTYPE, device="cuda")
    model.load_checkpoint(str(CKPT))
    model = model.to("cuda").eval()
    print(f"dtype={DTYPE}  model loaded: {time.time()-t0:.1f}s  "
          f"params={sum(p.numel() for p in model.parameters())/1e9:.2f}B")
    if torch.cuda.is_available():
        print(f"weights VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB\n")

    image = (torch.rand(1, 3, height, width) * 2.0 - 1.0)
    img_dev = image.to("cuda", dtype=model.torch_dtype)
    proprio = torch.zeros(1, proprio_dim)

    def enc_text():
        with torch.no_grad():
            model.encode_prompt(PROMPT)

    def enc_vae():
        with torch.no_grad():
            model._encode_input_image_latents_tensor(input_image=img_dev, tiled=False)

    def infer(steps):
        with torch.no_grad():
            return model.infer_action(prompt=PROMPT, input_image=image, action_horizon=action_horizon,
                                      proprio=proprio, num_inference_steps=steps, seed=0, rand_device="cpu")

    backends = {
        "MATH":    sdpa_kernel([SDPBackend.MATH]),
        "FLASH":   sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]),
        "default": nullcontext(),
    }

    print(f"{'backend':>8} | {'text(T5)':>9} | {'vision(VAE)':>11} | {'world-prefill':>13} | "
          f"{'plan/step':>9} | {'plan(20)':>9} | {'TOTAL(20)':>9}")
    print("-" * 92)
    results = {}
    for label, ctx in backends.items():
        try:
            with ctx:
                t_text = _timed(enc_text)
                t_vae = _timed(enc_vae)
                infer(1)  # warm
                t1 = _timed(lambda: infer(1))
                t20 = _timed(lambda: infer(NUM_STEPS))
            per_step = (t20 - t1) / (NUM_STEPS - 1)
            prefill = max(t1 - t_text - t_vae - per_step, 0.0)
            plan = per_step * NUM_STEPS
            results[label] = (t_text, t_vae, prefill, per_step, plan, t20)
            print(f"{label:>8} | {t_text:8.0f}m | {t_vae:10.0f}m | {prefill:12.0f}m | "
                  f"{per_step:8.0f}m | {plan:8.0f}m | {t20:8.0f}m")
        except Exception as e:
            print(f"{label:>8} | FAILED -> {type(e).__name__}: {str(e)[:80]}")

    # Correctness sanity on default backend.
    out = infer(NUM_STEPS)
    a = out["action"].detach().float().cpu().numpy()
    a = a[0] if (a.ndim == 3 and a.shape[0] == 1) else a
    assert a.shape[-1] == int(cfg.data.train.processor.action_output_dim) and np.isfinite(a).all()

    if "default" in results and "MATH" in results:
        speedup = results["MATH"][5] / results["default"][5]
        print(f"\nflash/default vs math total speedup: {speedup:.2f}x")
    print("PASS: planning benchmark complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
