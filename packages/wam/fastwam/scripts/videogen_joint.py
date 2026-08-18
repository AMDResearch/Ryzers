# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Open-loop imagined-video generation for FastWAM on Strix Halo (gfx1151).

Runs the joint video+action denoising path (`FastWAM.infer_joint`) from a single
ground-truth start frame + proprio + language prompt, at the standard FastWAM
setting (20 inference steps, no step reduction). Decodes the imagined future clip
and writes a two-column MP4: ground-truth future (left) vs FastWAM imagined
(right), per workspace rule 2.a. Reports the joint-path latency (video+action)
per clip and in aggregate.

Reuses the upstream `RobotVideoDataset` + `FastWAMProcessor` so cam-concat,
resize/crop, [-1,1] normalization and proprio normalization match training.

Env: FASTWAM_REPO, CONFIG_NAME, CKPT, DATASET_STATS, DATASET_DIR, NUM_VIDEOS(10),
     NUM_STEPS(20), OUT_DIR, TAG, SEED, FPS(6).
"""
import os
import sys
import json
import time

import numpy as np
import torch
import imageio
from PIL import Image, ImageDraw

FASTWAM_REPO = os.environ.get("FASTWAM_REPO", "/repos/fastwam")
CONFIG_NAME = os.environ.get("CONFIG_NAME", "sim_libero")
CKPT = os.environ["CKPT"]
DATASET_STATS = os.environ["DATASET_STATS"]
DATASET_DIR = os.environ["DATASET_DIR"]
NUM_VIDEOS = int(os.environ.get("NUM_VIDEOS") or "10")
NUM_STEPS = int(os.environ.get("NUM_STEPS") or "20")
OUT_DIR = os.environ.get("OUT_DIR", "/outputs")
TAG = os.environ.get("TAG", CONFIG_NAME)
SEED = int(os.environ.get("SEED") or "0")
FPS = int(os.environ.get("FPS") or "6")


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


def _build_dataset(cfg):
    from hydra.utils import instantiate
    import fastwam.datasets.lerobot.robot_video_dataset as rvd
    from fastwam.utils import misc

    def _stub_text_context(self, prompt):
        return torch.zeros(self.context_len, 8), torch.ones(self.context_len, dtype=torch.bool)
    rvd.RobotVideoDataset._get_cached_text_context = _stub_text_context
    try:
        misc.get_work_dir = lambda *a, **k: "/tmp"
    except Exception:
        pass

    return instantiate(
        cfg.data.train,
        dataset_dirs=[DATASET_DIR],
        is_training_set=False,
        val_set_proportion=0.0,
        pretrained_norm_stats=DATASET_STATS,
        skip_padding_as_possible=False,
    )


def _video_tensor_to_frames(video):
    """[C, T, H, W] in [-1,1] -> list of uint8 HxWx3 numpy frames."""
    v = video.detach().float().clamp(-1, 1)
    v = ((v + 1.0) * 127.5).to(torch.uint8).cpu().numpy()   # [C, T, H, W]
    return [np.ascontiguousarray(v[:, t].transpose(1, 2, 0)) for t in range(v.shape[1])]


def _to_rgb(frame):
    if isinstance(frame, Image.Image):
        return np.array(frame.convert("RGB"))
    return np.asarray(frame)[..., :3]


def _label(img, text):
    pil = Image.fromarray(img.astype(np.uint8))
    ImageDraw.Draw(pil).text((6, 6), text, fill=(255, 255, 0))
    return np.array(pil)


def main() -> int:
    print(f"torch {torch.__version__}  device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    print(f"config={CONFIG_NAME}  ckpt={os.path.basename(CKPT)}  videos={NUM_VIDEOS}  steps={NUM_STEPS}")

    if FASTWAM_REPO not in sys.path:
        sys.path.insert(0, FASTWAM_REPO)
    from hydra.utils import instantiate
    cfg = _compose_cfg()
    num_video_frames = int(cfg.data.train.num_frames)
    action_horizon = num_video_frames - 1
    print(f"num_video_frames={num_video_frames}  action_horizon={action_horizon}")

    t0 = time.time()
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda")
    model.load_checkpoint(str(CKPT))
    model = model.to("cuda").eval()
    print(f"model loaded: {time.time()-t0:.1f}s  proprio_dim={model.proprio_dim}")

    ds = _build_dataset(cfg)
    starts = ds.lerobot_dataset.episode_data_index["from"].tolist()
    n = min(NUM_VIDEOS, len(starts))
    out_dir = os.path.join(OUT_DIR, f"videogen_{TAG}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"episodes={len(starts)}  generating {n} imagined clips -> {out_dir}\n")

    per = []
    for k in range(n):
        idx = int(starts[k])
        sample = ds[idx]
        video = sample["video"]                                   # [C, T, H, W] in [-1,1]
        input_image = video[:, 0].unsqueeze(0).to("cuda", dtype=model.torch_dtype)
        proprio = sample["proprio"][0:1].to("cuda", dtype=model.torch_dtype)
        prompt = sample["prompt"]

        t1 = time.time()
        with torch.no_grad():
            out = model.infer_joint(
                prompt=prompt, input_image=input_image,
                num_video_frames=num_video_frames, action_horizon=action_horizon,
                proprio=proprio, num_inference_steps=NUM_STEPS, seed=SEED,
                rand_device="cpu", test_action_with_infer_action=False,
            )
        latency = time.time() - t1

        gen_frames = [_to_rgb(f) for f in out["video"]]           # list of HxWx3 uint8
        gt_frames = _video_tensor_to_frames(video)                # aligned GT future
        tt = min(len(gen_frames), len(gt_frames))

        stitched = []
        for gt, gen in zip(gt_frames[:tt], gen_frames[:tt]):
            if gt.shape[:2] != gen.shape[:2]:
                gt = np.array(Image.fromarray(gt).resize((gen.shape[1], gen.shape[0]), Image.BILINEAR))
            left = _label(gt, "GT")
            right = _label(gen, "FastWAM imagined")
            stitched.append(np.concatenate([left, right], axis=1))

        mp4 = os.path.join(out_dir, f"clip{k:02d}_gt_vs_imagined.mp4")
        imageio.mimwrite(mp4, stitched, fps=FPS, quality=8, macro_block_size=1)

        # action-error sanity (normalized space) vs GT
        gt_a = sample["action"].float().cpu().numpy()
        pr_a = out["action"].float().cpu().numpy()
        T = min(gt_a.shape[0], pr_a.shape[0])
        norm_mae = float(np.abs(pr_a[:T] - gt_a[:T]).mean())
        per.append({"clip": k, "frame_idx": idx, "frames": tt,
                    "joint_latency_s": round(latency, 3),
                    "action_norm_mae": round(norm_mae, 4), "prompt": prompt[:80]})
        print(f"clip{k:02d} idx={idx:7d} frames={tt} joint_latency={latency:.2f}s "
              f"actMAE={norm_mae:.4f} -> {os.path.basename(mp4)}")

    lat = np.array([p["joint_latency_s"] for p in per])
    # exclude first (warmup) from steady-state summary if >1 clip
    steady = lat[1:] if len(lat) > 1 else lat
    summary = {
        "tag": TAG, "config": CONFIG_NAME, "num_videos": n,
        "num_inference_steps": NUM_STEPS, "num_video_frames": num_video_frames,
        "fps": FPS,
        "joint_latency_s_mean_all": round(float(lat.mean()), 3),
        "joint_latency_s_mean_steady": round(float(steady.mean()), 3),
        "joint_latency_s_first_warmup": round(float(lat[0]), 3),
        "clips": per,
    }
    with open(os.path.join(out_dir, "video_latency.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n== {TAG} imagined-video generation ==")
    print(f"joint-path latency (video+action, {NUM_STEPS} steps): "
          f"warmup={lat[0]:.2f}s  steady-mean={steady.mean():.2f}s")
    print(f"videos + video_latency.json -> {out_dir}")
    print("PASS: video generation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
