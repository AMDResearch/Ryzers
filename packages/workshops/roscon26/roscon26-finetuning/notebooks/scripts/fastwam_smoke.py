# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Full-model GPU smoke for FastWAM (Wan2.2-TI2V-5B) on AMD Strix Halo (gfx1151).

Builds the real FastWAM model via the upstream hydra config, loads the released bf16 LIBERO
checkpoint, and runs ONE fast-route `infer_action` on a synthetic observation to prove the whole
flow-matching action path executes end-to-end on ROCm. No simulator/dataset needed. Reports
first-call (one-time ROCm kernel-JIT + autotune warmup) and steady-state latency + peak VRAM.
Exits non-zero on any failure so the ORGANIZER offline check / CI catches a broken image.

Env (baked-image defaults): FASTWAM_REPO (/repos/fastwam), CONFIG_NAME (sim_libero),
CKPT ($FASTWAM_RELEASE_DIR/libero_uncond_2cam224.pt), NUM_STEPS (20), PROMPT.
Run with the fastwam venv:  /opt/fastwam-venv/bin/python scripts/fastwam_smoke.py
"""
import os
import sys
import time

import numpy as np
import torch

FASTWAM_REPO = os.environ.get("FASTWAM_REPO", "/repos/fastwam")
RELEASE_DIR = os.environ.get("FASTWAM_RELEASE_DIR", "/opt/fastwam-assets/fastwam_release")
CKPT = os.environ.get("CKPT") or os.path.join(RELEASE_DIR, "libero_uncond_2cam224.pt")
CONFIG_NAME = os.environ.get("CONFIG_NAME", "sim_libero")
NUM_STEPS = int(os.environ.get("NUM_STEPS") or "20")  # upstream infer_action default
PROMPT = os.environ.get("PROMPT", "pick up the object and place it")


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
    print(f"torch            : {torch.__version__}  hip={torch.version.hip}")
    if not torch.version.hip:
        print("FAIL: torch is not a ROCm build.", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("FAIL: no ROCm device visible (check /dev/kfd, /dev/dri).", file=sys.stderr)
        return 1
    print(f"device[0]        : {torch.cuda.get_device_name(0)}")

    if FASTWAM_REPO not in sys.path:
        sys.path.insert(0, FASTWAM_REPO)
    if not os.path.exists(CKPT):
        print(f"FAIL: checkpoint not found: {CKPT}\n"
              "      run notebook 3's setup cell to download the public FastWAM assets,\n"
              "      or set FASTWAM_RELEASE_DIR to an existing cache.", file=sys.stderr)
        return 1

    from hydra.utils import instantiate
    cfg = _compose_cfg()

    video_size = list(cfg.data.train.video_size)
    height, width = int(video_size[0]), int(video_size[1])
    num_frames = int(cfg.data.train.num_frames)
    action_horizon = num_frames - 1
    proprio_dim = int(cfg.data.train.processor.proprio_output_dim)
    action_dim = int(cfg.data.train.processor.action_output_dim)
    print(f"config           : {CONFIG_NAME}  HxW={height}x{width}  "
          f"action_horizon={action_horizon}  proprio_dim={proprio_dim}  action_dim={action_dim}")

    t0 = time.time()
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda")
    model.load_checkpoint(str(CKPT))
    model = model.to("cuda").eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model loaded     : {time.time() - t0:.1f}s  params={n_params / 1e9:.2f}B")

    image = (torch.rand(1, 3, height, width) * 2.0 - 1.0)
    proprio = torch.zeros(1, proprio_dim)

    def _run():
        with torch.no_grad():
            return model.infer_action(
                prompt=PROMPT, input_image=image, action_horizon=action_horizon,
                proprio=proprio, num_inference_steps=NUM_STEPS, seed=0, rand_device="cpu",
            )

    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    _sync(); t0 = time.time(); _run(); _sync()
    cold_ms = (time.time() - t0) * 1000.0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _sync(); t1 = time.time(); out = _run(); _sync()
    warm_ms = (time.time() - t1) * 1000.0

    action = out["action"].detach().to(dtype=torch.float32, device="cpu").numpy()
    if action.ndim == 3 and action.shape[0] == 1:
        action = action[0]
    peak_gb = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
    print(f"infer_action     : {action.shape}  (steps={NUM_STEPS})")
    print(f"  first-call     : {cold_ms:.0f} ms  (incl. one-time ROCm warmup)")
    print(f"  steady-state   : {warm_ms:.0f} ms  peak={peak_gb:.1f} GB")

    if action.ndim != 2 or action.shape[-1] != action_dim:
        print(f"FAIL: expected (T, {action_dim}) action chunk, got {action.shape}", file=sys.stderr)
        return 1
    if not np.isfinite(action).all():
        print("FAIL: non-finite actions.", file=sys.stderr)
        return 1

    print("PASS: FastWAM full-model ROCm smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
