# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""FastWAM video imagination (joint path) for notebook 3 - FULL EPISODES, two different tasks.

Runs FastWAM's joint video+action denoising route (`FastWAM.infer_joint`) as a receding-horizon
rollout over the WHOLE episode (not a single 32-step chunk), for TWO genuinely different LIBERO
tasks, then writes one two-column MP4 per episode: ground-truth video (left) vs FastWAM imagined
(right) - the "what the model dreams" view (workshop rule 2.a: GT left, prediction right).

How a full episode is imagined
------------------------------
Each `infer_joint` call conditions on ONE start frame + ONE proprio vector and emits
`num_video_frames` video frames spanning one `action_horizon` (= num_frames-1) worth of actions,
sub-sampled by `action_video_freq_ratio` (so 33-frame / 32-action horizons -> 9 video frames). To
cover a full episode (~120-250 steps) we step the horizon window forward by `action_horizon` steps
and re-anchor each window on the real observation at that step - exactly the receding-horizon loop
the shipped LIBERO eval uses - concatenating the imagined segments into one full-length clip. This
re-grounds every horizon on ground truth (no unbounded drift) while still showing the model's
imagination across the entire task.

Two different tasks
-------------------
The baked dataset is LIBERO-Object (10 "pick up the <object> and place it in the basket" tasks). We
default to two visually/semantically distinct ones - the BBQ-sauce pick and the milk pick - selected
by their first episode index. Override with `EPISODES=<e0>,<e1>,...`.

This is a subprocess backend for the notebook: it runs in the isolated /opt/fastwam-venv and prints
one `VIDEOGEN_MP4=<path>` line per finished EPISODE plus a final `VIDEOGEN_SUMMARY=<json>` line the
notebook parses to embed the clips inline.

Env (all have baked-image defaults):
  FASTWAM_REPO   /repos/fastwam            CONFIG_NAME    sim_libero
  CKPT           $FASTWAM_RELEASE_DIR/libero_uncond_2cam224.pt
  DATASET_STATS  $FASTWAM_RELEASE_DIR/libero_uncond_2cam224_dataset_stats.json
  DATASET_DIR    $FASTWAM_DATA_DIR/libero_object_no_noops_lerobot
  EPISODES       "47,273"  (bbq sauce, milk)   NUM_STEPS 20   SEED 0   FPS 6
  MAX_STEPS_PER_EPISODE 0 (0 = full episode)   OUT_DIR /outputs   TAG libero_object
"""
import json
import os
import sys
import time

import numpy as np
import torch
import imageio
from PIL import Image, ImageDraw

FASTWAM_REPO = os.environ.get("FASTWAM_REPO", "/repos/fastwam")
CONFIG_NAME = os.environ.get("CONFIG_NAME", "sim_libero")
RELEASE_DIR = os.environ.get("FASTWAM_RELEASE_DIR", "/opt/fastwam-assets/fastwam_release")
DATA_DIR_BASE = os.environ.get("FASTWAM_DATA_DIR", "/opt/fastwam-assets/data")
CKPT = os.environ.get("CKPT") or os.path.join(RELEASE_DIR, "libero_uncond_2cam224.pt")
DATASET_STATS = os.environ.get("DATASET_STATS") or os.path.join(
    RELEASE_DIR, "libero_uncond_2cam224_dataset_stats.json")
DATASET_DIR = os.environ.get("DATASET_DIR") or os.path.join(
    DATA_DIR_BASE, "libero_object_no_noops_lerobot")
# Two DIFFERENT tasks by first-episode index: bbq sauce (ep47), milk (ep273). Override with EPISODES.
EPISODES = os.environ.get("EPISODES", "47,273")
NUM_STEPS = int(os.environ.get("NUM_STEPS") or "20")
SEED = int(os.environ.get("SEED") or "0")
FPS = int(os.environ.get("FPS") or "6")
MAX_STEPS_PER_EPISODE = int(os.environ.get("MAX_STEPS_PER_EPISODE") or "0")  # 0 = full episode
OUT_DIR = os.environ.get("OUT_DIR", "/outputs")
TAG = os.environ.get("TAG", "libero_object")


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


def _episode_tasks():
    """episode_index -> task string, from the dataset's meta/episodes.jsonl (if present)."""
    out = {}
    p = os.path.join(DATASET_DIR, "meta", "episodes.jsonl")
    if os.path.exists(p):
        for line in open(p):
            try:
                d = json.loads(line)
            except Exception:
                continue
            t = d.get("tasks") or d.get("task")
            if isinstance(t, list):
                t = t[0] if t else None
            out[int(d["episode_index"])] = t
    return out


def _slug(text, n=40):
    keep = "".join(c if (c.isalnum() or c == " ") else " " for c in (text or "clip"))
    return "_".join(keep.split())[:n] or "clip"


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


def _stitch(gt, gen):
    if gt.shape[:2] != gen.shape[:2]:
        gt = np.array(Image.fromarray(gt).resize((gen.shape[1], gen.shape[0]), Image.BILINEAR))
    left = _label(gt, "ground truth")
    right = _label(gen, "FastWAM imagined")
    return np.concatenate([left, right], axis=1)


def _rollout_episode(model, ds, f0, L, num_video_frames, action_horizon, ratio):
    """Receding-horizon imagination over one full episode.

    Steps the horizon window forward by `action_horizon` actions, re-anchoring each window on the
    ground-truth observation at that step. Returns (stitched_frames, action_mae, latencies, prompt).
    """
    stitched, gt_all_a, pr_all_a, lats = [], [], [], []
    t = 0
    prompt = None
    while t < L:
        sample = ds[int(f0 + t)]
        if prompt is None:
            prompt = sample["prompt"]
        gt_frames = _video_tensor_to_frames(sample["video"])        # NVF GT frames for this horizon
        input_image = sample["video"][:, 0].unsqueeze(0).to("cuda", dtype=model.torch_dtype)
        proprio = sample["proprio"][0:1].to("cuda", dtype=model.torch_dtype)

        t1 = time.time()
        with torch.no_grad():
            out = model.infer_joint(
                prompt=prompt, input_image=input_image,
                num_video_frames=num_video_frames, action_horizon=action_horizon,
                proprio=proprio, num_inference_steps=NUM_STEPS, seed=SEED,
                rand_device="cpu", test_action_with_infer_action=False,
            )
        lat = time.time() - t1
        lats.append(lat)
        gen_frames = [_to_rgb(f) for f in out["video"]]

        is_last = (t + action_horizon) >= L
        if is_last:
            ka = L - t                                              # actions remaining
            kv = min(num_video_frames, (ka + ratio - 1) // ratio)  # video frames covering them
        else:
            ka = action_horizon                                    # non-overlapping stride
            kv = num_video_frames - 1                              # drop the shared boundary frame

        tt = min(kv, len(gt_frames), len(gen_frames))
        for i in range(tt):
            stitched.append(_stitch(gt_frames[i], gen_frames[i]))

        gt_a = sample["action"].float().cpu().numpy()
        pr_a = out["action"].float().cpu().numpy()
        na = min(ka, gt_a.shape[0], pr_a.shape[0])
        gt_all_a.append(gt_a[:na])
        pr_all_a.append(pr_a[:na])

        win = t // action_horizon + 1
        nwin = (L + action_horizon - 1) // action_horizon
        print(f"    [window {win}/{nwin}] steps {t:3d}-{min(t+action_horizon, L):3d}  "
              f"frames={tt}  joint_latency={lat:.2f}s", flush=True)
        t += action_horizon

    gt_cat = np.concatenate(gt_all_a, axis=0) if gt_all_a else np.zeros((1, 1))
    pr_cat = np.concatenate(pr_all_a, axis=0) if pr_all_a else np.zeros((1, 1))
    mae = float(np.abs(pr_cat - gt_cat).mean())
    return stitched, mae, lats, prompt


def main() -> int:
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__}  hip={torch.version.hip}  device={dev}", flush=True)
    for p, what in ((CKPT, "checkpoint"), (DATASET_STATS, "dataset stats"), (DATASET_DIR, "GT dataset")):
        if not os.path.exists(p):
            print(f"FAIL: missing {what}: {p}", file=sys.stderr)
            return 1

    if FASTWAM_REPO not in sys.path:
        sys.path.insert(0, FASTWAM_REPO)
    from hydra.utils import instantiate
    cfg = _compose_cfg()
    num_frames = int(cfg.data.train.num_frames)
    ratio = int(cfg.data.train.action_video_freq_ratio)
    action_horizon = num_frames - 1
    num_video_frames = (num_frames - 1) // ratio + 1     # eval-aligned: 33/4 -> 9 video frames
    print(f"num_frames={num_frames}  action_horizon={action_horizon}  "
          f"num_video_frames={num_video_frames}  ratio={ratio}  steps={NUM_STEPS}", flush=True)

    t0 = time.time()
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda")
    model.load_checkpoint(str(CKPT))
    model = model.to("cuda").eval()
    print(f"model loaded: {time.time()-t0:.1f}s  proprio_dim={model.proprio_dim}", flush=True)

    ds = _build_dataset(cfg)
    edi = ds.lerobot_dataset.episode_data_index
    frm = edi["from"].tolist()
    to = edi["to"].tolist()
    tasks = _episode_tasks()
    n_ep = len(frm)

    want = [int(x) for x in EPISODES.split(",") if x.strip() != ""]
    want = [e for e in want if 0 <= e < n_ep]
    if not want:
        want = [0, min(273, n_ep - 1)]

    out_dir = os.path.join(OUT_DIR, f"videogen_{TAG}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"episodes in dataset={n_ep}  rendering FULL episodes for tasks: "
          f"{[e for e in want]} -> {out_dir}", flush=True)
    print("(joint route: first window JIT-warms ROCm kernels - can take several minutes)\n", flush=True)

    per = []
    all_lats = []
    for k, e in enumerate(want):
        f0, f1 = int(frm[e]), int(to[e])
        L = f1 - f0
        if MAX_STEPS_PER_EPISODE > 0:
            L = min(L, MAX_STEPS_PER_EPISODE)
        task = tasks.get(e) or f"episode {e}"
        print(f"[{k+1}/{len(want)}] episode {e}  task={task!r}  steps={L}", flush=True)

        stitched, mae, lats, prompt = _rollout_episode(
            model, ds, f0, L, num_video_frames, action_horizon, ratio)
        all_lats.extend(lats)

        mp4 = os.path.join(out_dir, f"episode{e:03d}_{_slug(task)}_gt_vs_imagined.mp4")
        imageio.mimwrite(mp4, stitched, fps=FPS, quality=8, macro_block_size=1)
        ep_lat = float(np.sum(lats))
        per.append({"episode": e, "task": task, "frames": len(stitched),
                    "windows": len(lats), "episode_latency_s": round(ep_lat, 2),
                    "action_norm_mae": round(mae, 4), "mp4": mp4})
        print(f"  -> {len(stitched)} frames, {len(lats)} windows, "
              f"episode joint-time={ep_lat:.1f}s, actMAE={mae:.4f}\n", flush=True)
        print(f"VIDEOGEN_MP4={mp4}", flush=True)

    lat = np.array(all_lats) if all_lats else np.array([0.0])
    steady = lat[1:] if len(lat) > 1 else lat
    summary = {
        "tag": TAG, "config": CONFIG_NAME, "num_videos": len(per),
        "num_inference_steps": NUM_STEPS, "num_video_frames": num_video_frames, "fps": FPS,
        "joint_latency_s_first_warmup": round(float(lat[0]), 3),
        "joint_latency_s_mean_steady": round(float(steady.mean()), 3),
        "clips": per,
    }
    with open(os.path.join(out_dir, "video_latency.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n== {TAG} full-episode imagination ==", flush=True)
    print(f"per-window joint latency ({NUM_STEPS} steps): warmup={lat[0]:.2f}s  "
          f"steady-mean={steady.mean():.2f}s", flush=True)
    print("VIDEOGEN_SUMMARY=" + json.dumps(summary), flush=True)
    print("PASS: FastWAM full-episode video imagination complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
