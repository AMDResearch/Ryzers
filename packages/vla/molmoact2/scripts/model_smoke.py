# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Capability 1: full-model smoke test for MolmoAct2 on AMD Strix Halo (gfx1151).

Initializes the *real* MolmoAct2-DROID checkpoint (downloads ~22 GB once into the
mounted HF cache), then runs a single action-prediction on a dummy observation to
prove the whole flow-matching action path executes end-to-end on ROCm. Reuses the
upstream `Policy` loader (bf16 dtype patches) baked at /repos/molmoact2/examples/droid.

Env: MODEL_REPO (allenai/MolmoAct2-DROID), DTYPE (bfloat16), NUM_STEPS (10).
Exits non-zero on any failure so `ryzers run` / CI catches a broken image.
"""
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

MODEL_REPO = os.environ.get("MODEL_REPO", "allenai/MolmoAct2-DROID")
SERVER_DIR = os.environ.get("DROID_SERVER_DIR", "/repos/molmoact2/examples/droid")
DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
    os.environ.get("DTYPE", "bfloat16")]
NUM_STEPS = int(os.environ.get("NUM_STEPS") or "10")


def predict_chunk(policy, norm_tag, images, task, state, num_steps):
    """Call the model's flow-matching action head, tolerating the
    inference_action_mode (new) vs action_mode (old) kwarg rename."""
    kw = dict(processor=policy.processor, images=images, task=task, state=state,
              norm_tag=norm_tag, enable_depth_reasoning=False, num_steps=num_steps,
              normalize_language=True, enable_cuda_graph=False)
    try:
        out = policy.model.predict_action(inference_action_mode="continuous", **kw)
    except TypeError as e:
        if "action_mode" not in str(e):
            raise
        out = policy.model.predict_action(action_mode="continuous", **kw)
    raw = out.actions if hasattr(out, "actions") else out
    if torch.is_tensor(raw):
        raw = raw.detach().to(dtype=torch.float32, device="cpu").numpy()
    a = np.asarray(raw, dtype=np.float32)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    return a


def main() -> int:
    print(f"torch            : {torch.__version__}  hip={torch.version.hip}")
    if not torch.version.hip:
        print("FAIL: torch is not a ROCm build.", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("FAIL: no ROCm device visible (check /dev/kfd, /dev/dri).",
              file=sys.stderr)
        return 1
    print(f"device[0]        : {torch.cuda.get_device_name(0)}")

    sys.path.insert(0, SERVER_DIR)
    from host_server_droid import Policy, NORM_TAG, REPO_ID  # noqa: E402

    repo = MODEL_REPO or REPO_ID
    print(f"loading model    : {repo}  (dtype={DTYPE}); first run downloads ~22 GB")
    t0 = time.time()
    policy = Policy(repo_id=repo, device="cuda:0", dtype=DTYPE)
    n_params = sum(p.numel() for p in policy.model.parameters())
    print(f"model loaded     : {time.time()-t0:.1f}s  params={n_params/1e9:.2f}B  norm_tag={NORM_TAG}")

    # One dummy DROID observation: 3 cams (ext1, ext2, wrist) + 8-DoF state.
    imgs = [Image.fromarray(np.random.randint(0, 255, (180, 320, 3), dtype=np.uint8))
            for _ in range(3)]
    state = np.zeros(8, dtype=np.float32)
    t1 = time.time()
    actions = predict_chunk(policy, NORM_TAG, imgs, "pick up the object", state, NUM_STEPS)
    dt = (time.time() - t1) * 1000.0
    print(f"predict_action   : {actions.shape}  in {dt:.0f} ms")
    if actions.ndim != 2 or actions.shape[-1] != 8:
        print(f"FAIL: expected (N, 8) action chunk, got {actions.shape}", file=sys.stderr)
        return 1
    if not np.isfinite(actions).all():
        print("FAIL: non-finite actions.", file=sys.stderr)
        return 1
    print("PASS: MolmoAct2 full-model ROCm smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
