# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Capability 2: DROID open-loop replay demo for MolmoAct2 on Strix Halo (gfx1151).

Picks a random episode from the real allenai/MolmoAct2-DROID-Dataset (override with
EPISODE=<n>), runs the MolmoAct2-DROID policy open-loop in a receding-horizon
fashion (replan every STRIDE steps), and writes two artifacts under $OUT_DIR:

  * droid_ep<E>.mp4              - the episode's exterior camera view (the scene the
                                   model predicts on)
  * droid_ep<E>_actions.png     - GT (teleop) vs predicted 8-DoF action trajectory,
                                   overlaid on the same axes per dim (workspace
                                   rule 2.a: numeric data with ground truth ->
                                   overlay GT + prediction)

This is a PORT-FIDELITY check (L1/MSE vs teleop GT), not the authors' headline
metric, which is closed-loop task success. See the LIBERO demo.

Env: EVAL_REPO, MODEL_REPO, EPISODE (random if unset), SEED, STRIDE (15),
DTYPE (bfloat16), NUM_STEPS (10), OUT_DIR (/outputs), CAM (exterior_1_left).
"""
import os
import random
import sys
import time

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

EVAL_REPO = os.environ.get("EVAL_REPO", "allenai/MolmoAct2-DROID-Dataset")
MODEL_REPO = os.environ.get("MODEL_REPO", "allenai/MolmoAct2-DROID")
SERVER_DIR = os.environ.get("DROID_SERVER_DIR", "/repos/molmoact2/examples/droid")
STRIDE = int(os.environ.get("STRIDE") or "15")
NUM_STEPS = int(os.environ.get("NUM_STEPS") or "10")
DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
    os.environ.get("DTYPE", "bfloat16")]
OUT_DIR = os.environ.get("OUT_DIR", "/outputs")
VIEW_CAM = "observation.images." + os.environ.get("CAM", "exterior_1_left")
CAMS = ["observation.images.exterior_1_left",
        "observation.images.exterior_2_left",
        "observation.images.wrist_left"]
DIM_NAMES = ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5",
             "joint_6", "gripper"]


def predict_chunk(policy, norm_tag, images, task, state, num_steps):
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


def decode_clip(path, start_idx, n_frames, rate, tb, stream, container):
    """Decode `n_frames` contiguous frames starting at source frame start_idx."""
    seek_pts = int((start_idx / rate) / tb)
    try:
        container.seek(max(seek_pts, 0), stream=stream, backward=True, any_frame=False)
    except Exception:  # noqa: BLE001
        container.seek(0)
    frames = []
    for frame in container.decode(stream):
        if frame.pts is None:
            continue
        idx = int(round(float(frame.pts * tb) * rate))
        if idx < start_idx:
            continue
        frames.append(frame.to_ndarray(format="rgb24"))
        if len(frames) >= n_frames:
            break
    return frames


def grab_frames(path, indices):
    import av
    want = sorted(set(int(i) for i in indices))
    out = {}
    if not want:
        return out
    container = av.open(path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    rate = float(stream.average_rate)
    tb = stream.time_base
    seek_pts = int((want[0] / rate) / tb)
    try:
        container.seek(max(seek_pts, 0), stream=stream, backward=True, any_frame=False)
    except Exception:  # noqa: BLE001
        container.seek(0)
    remaining = set(want)
    for frame in container.decode(stream):
        if frame.pts is None:
            continue
        idx = int(round(float(frame.pts * tb) * rate))
        if idx in remaining:
            out[idx] = frame.to_ndarray(format="rgb24")
            remaining.discard(idx)
        elif idx > want[-1]:
            break
        if not remaining:
            break
    container.close()
    if remaining and out:
        got = sorted(out)
        for idx in list(remaining):
            out[idx] = out[min(got, key=lambda g: abs(g - idx))]
    return out


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    seed_env = os.environ.get("SEED", "")
    if seed_env:
        random.seed(int(seed_env))
    print(f"torch {torch.__version__} hip={torch.version.hip} dev={torch.cuda.get_device_name(0)}")

    info = json_load(hf_hub_download(EVAL_REPO, "meta/info.json", repo_type="dataset"))
    fps = info["fps"]
    data_tmpl = info["data_path"]
    video_tmpl = info["video_path"]
    meta_ep = pq.read_table(hf_hub_download(
        EVAL_REPO, "meta/episodes/chunk-000/file-000.parquet", repo_type="dataset")).to_pydict()
    episodes = list(meta_ep["episode_index"])

    ep_env = os.environ.get("EPISODE", "")
    episode = int(ep_env) if ep_env else random.choice(episodes)
    mr = episodes.index(episode)
    print(f"episode          : {episode}  ({len(episodes)} available; "
          f"{'fixed' if ep_env else 'random'})")

    d_chunk, d_file = meta_ep["data/chunk_index"][mr], meta_ep["data/file_index"][mr]
    length = meta_ep["length"][mr]
    task = meta_ep["tasks"][mr]
    task = task[0] if isinstance(task, (list, tuple)) else task
    print(f"task             : {task!r}  length={length} frames")

    # dataset_from/to_index are GLOBAL row indices; filter the per-file table by
    # episode_index so this works for any episode regardless of which file it is in.
    dt = pq.read_table(hf_hub_download(
        EVAL_REPO, data_tmpl.format(chunk_index=d_chunk, file_index=d_file), repo_type="dataset"))
    ep_table = dt.filter(pc.equal(dt.column("episode_index"), episode))

    def col(name):
        return ep_table.column(name).to_pylist()

    states = np.asarray(col("observation.state"), dtype=np.float32)
    gt_actions = np.asarray(col("action"), dtype=np.float32)
    timestamps = np.asarray(col("timestamp"), dtype=np.float64)

    sys.path.insert(0, SERVER_DIR)
    from host_server_droid import Policy, NORM_TAG  # noqa: E402
    t0 = time.time()
    policy = Policy(repo_id=MODEL_REPO, device="cuda:0", dtype=DTYPE)
    print(f"model loaded     : {time.time()-t0:.1f}s "
          f"params={sum(p.numel() for p in policy.model.parameters())/1e9:.2f}B")

    # Frames at each replan point for all 3 input cams.
    replan_pts = list(range(0, length, STRIDE))
    cam_meta = {c: dict(chunk=meta_ep[f"videos/{c}/chunk_index"][mr],
                        file=meta_ep[f"videos/{c}/file_index"][mr],
                        from_ts=meta_ep[f"videos/{c}/from_timestamp"][mr]) for c in CAMS}
    frames_by_cam = {}
    for c in CAMS:
        cm = cam_meta[c]
        path = hf_hub_download(EVAL_REPO, video_tmpl.format(
            video_key=c, chunk_index=cm["chunk"], file_index=cm["file"]), repo_type="dataset")
        idxs = [int(round((cm["from_ts"] + timestamps[t]) * fps)) for t in replan_pts]
        grabbed = grab_frames(path, idxs)
        frames_by_cam[c] = {t: grabbed[int(round((cm["from_ts"] + timestamps[t]) * fps))]
                            for t in replan_pts}

    # Receding-horizon open-loop prediction -> dense predicted trajectory.
    pred = np.full_like(gt_actions, np.nan)
    lat = []
    for t in replan_pts:
        imgs = [Image.fromarray(frames_by_cam[c][t]) for c in CAMS]
        torch.cuda.synchronize()
        ts = time.perf_counter()
        chunk = predict_chunk(policy, NORM_TAG, imgs, task, states[t], NUM_STEPS)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - ts) * 1000.0)
        n = min(STRIDE, length - t, chunk.shape[0])
        pred[t:t + n] = chunk[:n]

    valid = ~np.isnan(pred).any(axis=1)
    l1 = float(np.abs(pred[valid] - gt_actions[valid]).mean())
    mse = float(((pred[valid] - gt_actions[valid]) ** 2).mean())
    print(f"open-loop        : L1={l1:.4f} MSE={mse:.4f}  ({np.mean(lat):.0f} ms/infer)")

    write_video(os.path.join(OUT_DIR, f"droid_ep{episode}.mp4"),
                cam_meta[VIEW_CAM], video_tmpl, length, timestamps, fps)
    plot_actions(os.path.join(OUT_DIR, f"droid_ep{episode}_actions.png"),
                 gt_actions, pred, episode, task, l1, mse)
    print(f"PASS: artifacts written to {OUT_DIR}")
    return 0


def write_video(out_path, cm, video_tmpl, length, timestamps, fps):
    import av
    import imageio
    path = hf_hub_download(EVAL_REPO, video_tmpl.format(
        video_key=VIEW_CAM, chunk_index=cm["chunk"], file_index=cm["file"]), repo_type="dataset")
    container = av.open(path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    rate = float(stream.average_rate)
    tb = stream.time_base
    start_idx = int(round((cm["from_ts"] + timestamps[0]) * fps))
    frames = decode_clip(path, start_idx, length, rate, tb, stream, container)
    container.close()
    if not frames:
        print("WARN: no frames decoded for video", file=sys.stderr)
        return
    imageio.mimsave(out_path, frames, fps=int(round(fps)), codec="libx264", quality=7)
    print(f"video            : {out_path}  ({len(frames)} frames @ {int(round(fps))} fps)")


def plot_actions(out_path, gt, pred, episode, task, l1, mse):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), squeeze=False)
    x = np.arange(gt.shape[0])
    for d in range(8):
        ax = axes[d // 4][d % 4]
        ax.plot(x, gt[:, d], color="tab:blue", lw=1.4, label="GT (teleop)")
        ax.plot(x, pred[:, d], color="tab:red", lw=1.2, ls="--", label="MolmoAct2 (pred)")
        ax.set_title(DIM_NAMES[d], fontsize=10)
        ax.tick_params(labelsize=7)
        if d == 0:
            ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"MolmoAct2-DROID open-loop (ROCm gfx1151), ep {episode}\n"
                 f"{task}\nL1={l1:.4f}  MSE={mse:.4f}  (GT solid, pred dashed)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"action plot      : {out_path}")


def json_load(path):
    import json
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    raise SystemExit(main())
