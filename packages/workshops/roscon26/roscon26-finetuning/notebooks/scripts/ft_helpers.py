# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Plumbing for the MolmoAct2 LoRA fine-tuning notebook.

The notebook keeps the *teaching* code inline (env-quieting, the exact train/eval
commands, the flow-matching action-head call). The bulky, non-teaching machinery lives
here so the cells stay readable:

  * subprocess streamers  - stream a child process into the notebook, preserving '\\r'
    so tqdm bars redraw in place; scrape (step, loss) points; hide chatty lines.
  * HF download machinery  - cache-aware prefetch with a GB heartbeat, local-asset
    staging, an input preflight, and the optional full-LIBERO route.
  * DROID video decode     - open-loop rollout on a few real DROID episodes, decoding
    only the frames we replan on and rendering GT-vs-predicted actions inline.
"""
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time

import numpy as np
import torch
from huggingface_hub import snapshot_download

# ---------------------------------------------------------------------------------------
# Subprocess streamers
# ---------------------------------------------------------------------------------------
# Backstop line filter: even with the notebook's warning-suppression flags, drop any stray
# ROCm/torch warning lines so the streamed subprocess output stays readable.
_NOISE = (
    "UserWarning",
    "HIPBLAS_STATUS",
    "hipblasLtMatmul",
    "Triggered internally",
    "scaled_dot_product_attention",
    "run_backward",
    "aotriton",
    "AOTRITON",
)

# LeRobot logs one metrics line per `log_freq` steps, e.g. "step:10 ... loss:1.684 grdn:.. lr:..".
# We scrape those (step, loss) points so Step 4 can draw a small loss curve. Handles the
# human-formatted step number (10, 1.5K, 2M) that LeRobot prints for large runs.
_STEP_RE = re.compile(r"\bstep:\s*([0-9.]+)\s*([KMB]?)")
_LOSS_RE = re.compile(r"\bloss:\s*([0-9.]+)")
_SUF = {"": 1.0, "K": 1e3, "M": 1e6, "B": 1e9}


def _popen(cmd, env=None, cwd=None):
    print("$ " + (cmd if isinstance(cmd, str) else " ".join(map(str, cmd))) + "\n", flush=True)
    # Binary pipe (no text=True): universal-newline mode would rewrite every '\r' to '\n' and turn
    # a single self-updating tqdm bar into one new line per tick. We decode + split ourselves.
    return subprocess.Popen(
        cmd,
        shell=isinstance(cmd, str),
        env=env,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )


def _pump(p, quiet=True, scrape=None, drop=None):
    """Stream a child process into the notebook, PRESERVING carriage returns so tqdm/progress bars
    redraw in place on ONE line (Jupyter honors '\\r'). `scrape(line)` collects data from complete
    '\\n' lines; `drop(line)` hides a complete line from the display (it is still scraped)."""
    import codecs

    dec = codecs.getincrementaldecoder("utf-8")("replace")
    buf = ""
    while True:
        chunk = p.stdout.read(4096)
        if not chunk:
            break
        buf += dec.decode(chunk)
        while True:
            i_n, i_r = buf.find("\n"), buf.find("\r")
            idxs = [i for i in (i_n, i_r) if i != -1]
            if not idxs:
                break
            cut = min(idxs) + 1
            seg, buf = buf[:cut], buf[cut:]
            if quiet and any(tok in seg for tok in _NOISE):
                continue
            full_line = seg.endswith("\n")
            if full_line and scrape:
                scrape(seg)
            if not (full_line and drop and drop(seg)):
                sys.stdout.write(seg)
                sys.stdout.flush()
    if buf and not (quiet and any(tok in buf for tok in _NOISE)):
        if scrape:
            scrape(buf)
        if not (drop and drop(buf)):
            sys.stdout.write(buf)
            sys.stdout.flush()
    p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"command failed (exit {p.returncode})")


def stream_cmd(cmd, env=None, cwd=None, quiet=True, scrape=None, drop=None):
    """Stream a subprocess into the notebook; tqdm/progress bars stay on ONE self-updating line.
    Optional `scrape`/`drop` callbacks collect from / hide complete lines (see `_pump`)."""
    _pump(_popen(cmd, env=env, cwd=cwd), quiet=quiet, scrape=scrape, drop=drop)


def run_cmd(cmd, env=None, cwd=None, quiet_warnings=True):
    """Stream a subprocess into the notebook; tqdm/progress bars stay on ONE self-updating line."""
    _pump(_popen(cmd, env=env, cwd=cwd), quiet=quiet_warnings)


def run_train(cmd, env=None, cwd=None):
    """Stream a training subprocess and return captured [(step, loss), ...]. The tqdm progress bar
    stays on one line; the chatty per-step "step:.. loss:.." metric lines are hidden from the
    display (still scraped, so the loss curve below is unaffected)."""
    pts = []

    def _scrape(line):
        ms, ml = _STEP_RE.search(line), _LOSS_RE.search(line)
        if ms and ml:
            pts.append((float(ms.group(1)) * _SUF[ms.group(2)], float(ml.group(1))))

    def _drop(line):
        return bool(_STEP_RE.search(line) and _LOSS_RE.search(line))

    _pump(_popen(cmd, env=env, cwd=cwd), quiet=True, scrape=_scrape, drop=_drop)
    return pts


# ---------------------------------------------------------------------------------------
# HF download machinery
# ---------------------------------------------------------------------------------------
def _hub():
    return os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")


def _repo_dir(repo_id, repo_type):
    pfx = "datasets--" if repo_type == "dataset" else "models--"
    return os.path.join(_hub(), pfx + repo_id.replace("/", "--"))


def _repair_dangling_hub_links():
    """Remove top-level cache links left by an unavailable legacy asset mount.

    Older workshop deployments linked whole Hub repositories in the persistent user home to
    `/opt/auplc-assets`. After switching to the online image, those links can outlive their mount.
    Hugging Face then cannot create the repository's `blobs/` or `trees/` directories because the
    repository path is a dangling symlink. Only broken top-level links are removed; valid staged
    repositories and Hugging Face's internal snapshot symlinks are untouched.
    """
    hub = _hub()
    if not os.path.isdir(hub):
        return
    for entry in os.scandir(hub):
        if entry.is_symlink() and not os.path.exists(entry.path):
            target = os.readlink(entry.path)
            os.unlink(entry.path)
            print(f"    REPAIR  removed stale HF cache link: {entry.name} -> {target}")


def _dir_gb(path):
    try:
        return int(subprocess.check_output(["du", "-sb", path], stderr=subprocess.DEVNULL).split()[0]) / 1e9
    except Exception:
        return 0.0


def _fully_cached(repo_id, repo_type, revision=None):
    # Hugging Face 1.x returns an existing snapshot directory in local-only mode even when it
    # contains just one previously downloaded file. Prefer its cached tree manifest, which lists
    # every file and expected size, so an interrupted/partial snapshot is never reported complete.
    repo_dir = _repo_dir(repo_id, repo_type)
    if revision:
        manifest = os.path.join(repo_dir, "trees", f"{revision}.json")
        if os.path.isfile(manifest):
            try:
                with open(manifest) as manifest_file:
                    files = json.load(manifest_file).get("files", {})
                snapshot = os.path.join(repo_dir, "snapshots", revision)
                return bool(files) and all(
                    os.path.isfile(os.path.join(snapshot, rel))
                    and os.path.getsize(os.path.join(snapshot, rel)) == meta["size"]
                    for rel, meta in files.items()
                )
            except (OSError, KeyError, TypeError, ValueError):
                return False

        # Legacy offline bundles predate the tree-manifest cache. They are complete by
        # construction and live under /opt; retain their local-only compatibility.
        if os.path.realpath(repo_dir).startswith("/opt/auplc-"):
            try:
                snapshot_download(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    revision=revision,
                    local_files_only=True,
                )
                return True
            except Exception:
                return False
        return False

    # Unpinned compatibility path: local-only is the best available check.
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            local_files_only=True,
        )
        return True
    except Exception:
        return False


def _heartbeat(repo_id, repo_type, stop):
    d, last = _repo_dir(repo_id, repo_type), _dir_gb(_repo_dir(repo_id, repo_type))
    while not stop.wait(3):
        cur = _dir_gb(d)
        print(f"      ...{repo_id}: {cur:.2f} GB on disk (+{max(cur - last, 0):.2f} GB/3s)", flush=True)
        last = cur


def prefetch(repo_id, repo_type, revision=None, tries=5):
    """Download `repo_id` into the HF cache (idempotent). Skips instantly if fully cached; shows a
    plain GB heartbeat while pulling (VERBOSE_DOWNLOAD=1); retries through network hiccups.
    Returns the resolved snapshot directory."""
    _repair_dangling_hub_links()
    verbose_dl = os.environ.get("VERBOSE_DOWNLOAD", "1") == "1"
    if _fully_cached(repo_id, repo_type, revision=revision):
        print(f"    CACHED  [{repo_type}] {repo_id}  ({_dir_gb(_repo_dir(repo_id, repo_type)):.2f} GB) - skipping")
        return snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            local_files_only=True,
        )
    print(f"    FETCH   [{repo_type}] {repo_id}  - downloading into HF cache", flush=True)
    for n in range(1, tries + 1):
        stop, th = threading.Event(), None
        if verbose_dl:
            th = threading.Thread(target=_heartbeat, args=(repo_id, repo_type, stop), daemon=True)
            th.start()
        try:
            t0 = time.time()
            snapshot = snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
            )
            stop.set()
            if th:
                th.join(timeout=1)
            print(f"    DONE    [{repo_type}] {repo_id}  ({_dir_gb(_repo_dir(repo_id, repo_type)):.2f} GB in {time.time() - t0:.0f}s)")
            return snapshot
        except Exception as e:
            stop.set()
            if th:
                th.join(timeout=1)
            print(f"      (network hiccup on {repo_id}, retry {n}/{tries}: {str(e)[:80]}; resuming cached bytes)")
            if n == tries:
                raise
            time.sleep(5)


def link_lerobot_dataset(repo_id, snapshot):
    """Expose an HF-cached dataset where LeRobot looks for local datasets.

    This prevents LeRobot from downloading a second copy after snapshot_download. It also keeps
    the old workshop subset layout working: whichever snapshot was resolved is the one training
    sees.
    """
    lr_home = os.environ.get(
        "HF_LEROBOT_HOME",
        os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "lerobot"),
    )
    dst = os.path.join(lr_home, *repo_id.split("/"))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        if os.path.islink(dst) and os.path.realpath(dst) == os.path.realpath(snapshot):
            print(f"    LeRobot dataset already linked -> {dst}")
            return dst
        if os.path.islink(dst) or os.path.isfile(dst):
            os.remove(dst)
        else:
            shutil.rmtree(dst)
    os.symlink(os.path.realpath(snapshot), dst)
    print(f"    LeRobot dataset link -> {dst}")
    return dst


def _stage_from_dir(assets_dir):
    """Copy a local resources dir (base ckpt + dataset under hf_hub/, plus the reference checkpoint)
    into the HF cache + REFERENCE_POLICY so prefetch() finds everything cached. Idempotent."""
    import shutil

    if not assets_dir or not os.path.isdir(assets_dir):
        return
    hub = _hub()
    hub_src = os.path.join(assets_dir, "hf_hub")
    if os.path.isdir(hub_src):
        os.makedirs(hub, exist_ok=True)
        for _name in sorted(os.listdir(hub_src)):
            _s, _d = os.path.join(hub_src, _name), os.path.join(hub, _name)
            if os.path.isdir(_s) and not os.path.exists(_d):
                print(f"    STAGE   {_name} -> HF cache", flush=True)
                shutil.copytree(_s, _d, symlinks=True)
    _ref_src = os.path.join(assets_dir, "checkpoints", "reference", "pretrained_model")
    _ref_dst = os.environ.get("REFERENCE_POLICY", os.path.expanduser("~/checkpoints/reference/pretrained_model"))
    if os.path.isdir(_ref_src) and not os.path.isdir(_ref_dst):
        print(f"    STAGE   reference checkpoint -> {_ref_dst}", flush=True)
        os.makedirs(os.path.dirname(_ref_dst), exist_ok=True)
        shutil.copytree(_ref_src, _ref_dst, symlinks=True)


def stage_assets():
    """If ASSETS_DIR is set, stage the base checkpoint + dataset + fine-tuned checkpoint from local
    storage into the HF cache (no Hub re-download). Otherwise use the configured cache and let
    prefetch() download any missing public files."""
    assets_dir = os.environ.get("ASSETS_DIR", "").strip()
    if assets_dir:
        print(f"== staging assets from ASSETS_DIR={assets_dir} (no Hub re-download) ==")
        _stage_from_dir(assets_dir)
        print("== asset staging done ==\n")
    else:
        print(f"== using HF cache: HF_HOME={os.environ.get('HF_HOME', '')} (missing public files will download) ==\n")


def preflight_inputs(base_ckpt, dataset_repo, base_revision=None, dataset_revision=None):
    """Confirm every input (base checkpoint, dataset, optional policy) is found BEFORE the long
    model load / training, so a missing asset fails fast with a clear message."""
    print("== preflight: inputs the notebook needs ==")
    _needed = [
        (base_ckpt, "model", base_revision),
        (dataset_repo, "dataset", dataset_revision),
    ]
    _pol = os.environ.get("POLICY_PATH") or os.environ.get("REFERENCE_POLICY", "")
    for _rid, _rt, _rev in _needed:
        _d = _repo_dir(_rid, _rt)
        if _fully_cached(_rid, _rt, revision=_rev):
            print(f"  [ok]      {_rt:7s} {_rid}  CACHED ({_dir_gb(_d):.2f} GB)")
        else:
            print(f"  [missing] {_rt:7s} {_rid}  will download")
    if _pol:
        _ok = os.path.isdir(_pol) and os.path.exists(os.path.join(_pol, "config.json"))
        print(f"  [{'ok' if _ok else 'n/a'}]      policy  {_pol}  {'found' if _ok else '(not staged yet)'}")
    print("== end preflight ==\n")


def fetch_full_libero(dataset_repo, revision=None):
    """EXTENDED ROUTE (USE_FULL_LIBERO=1): pull the COMPLETE LIBERO dataset from the Hub (needs
    internet; ~33 GB) and repoint the LeRobot dataset home at it, replacing the staged subset.
    Because the subset reuses each file's real content hash, only missing files download."""
    import shutil
    import huggingface_hub.constants as _hc

    print("USE_FULL_LIBERO=1 -> fetching the COMPLETE LIBERO dataset from the Hub (large; needs internet) ...", flush=True)
    _prev_off = _hc.HF_HUB_OFFLINE
    _hc.HF_HUB_OFFLINE = False  # this flag is captured at import time; flip it to actually reach the Hub
    os.environ["HF_HUB_OFFLINE"] = "0"
    try:
        _full = snapshot_download(
            repo_id=dataset_repo,
            repo_type="dataset",
            revision=revision,
            local_files_only=False,
        )
    finally:
        _hc.HF_HUB_OFFLINE = _prev_off
        os.environ["HF_HUB_OFFLINE"] = "1" if _prev_off else "0"
    # Point the LeRobot dataset home at the full snapshot (replaces the staged-subset link) so Step 4
    # trains on the complete dataset.
    _lr = link_lerobot_dataset(dataset_repo, _full)
    print(f"    full LIBERO dataset ready ({_dir_gb(_full):.1f} GB) -> {_lr}")
    return _full


def _safe_extract_tar(archive, destination):
    """Extract a public dataset archive without allowing links or paths outside destination."""
    root = os.path.realpath(destination)
    with tarfile.open(archive, "r:*") as tf:
        members = tf.getmembers()
        for member in members:
            resolved = os.path.realpath(os.path.join(root, member.name))
            if os.path.commonpath((root, resolved)) != root:
                raise RuntimeError(f"unsafe path in {archive}: {member.name}")
            if member.issym() or member.islnk() or member.isdev():
                raise RuntimeError(f"unsupported link/device in {archive}: {member.name}")
        tf.extractall(root, members=members)


def prepare_fastwam_assets(release_dir, diffsynth_dir, data_dir, include_dataset=True):
    """Download FastWAM's released public inputs from Hugging Face.

    Files are placed in the exact directory layout expected by the pinned FastWAM loader. Existing
    files are reused, so this is safe to run whenever notebook 3 starts.
    """
    release_dir = os.path.abspath(os.path.expanduser(release_dir))
    diffsynth_dir = os.path.abspath(os.path.expanduser(diffsynth_dir))
    data_dir = os.path.abspath(os.path.expanduser(data_dir))
    os.makedirs(release_dir, exist_ok=True)
    os.makedirs(diffsynth_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    release_files = [
        "libero_uncond_2cam224.pt",
        "libero_uncond_2cam224_dataset_stats.json",
    ]
    if not all(os.path.isfile(os.path.join(release_dir, name)) for name in release_files):
        print("== downloading public FastWAM checkpoint (~12 GB) ==")
        snapshot_download(
            repo_id="yuanty/fastwam",
            revision="8eaceeb24c3cc92ff2a9c9a9d266a4941b836705",
            allow_patterns=release_files,
            local_dir=release_dir,
        )
    else:
        print("== public FastWAM checkpoint already cached ==")

    redirect = os.environ.get("FASTWAM_REDIRECT_COMMON_FILES", "false").lower() == "true"
    if redirect:
        # Legacy offline-event bundles contain FastWAM's converted ModelScope files. They are not
        # fetched by the public path, but accepting them keeps those explicitly requested images
        # functional.
        converted = os.path.join(
            diffsynth_dir,
            "DiffSynth-Studio",
            "Wan-Series-Converted-Safetensors",
        )
        converted_files = [
            os.path.join(converted, "models_t5_umt5-xxl-enc-bf16.safetensors"),
            os.path.join(converted, "Wan2.2_VAE.safetensors"),
            os.path.join(
                diffsynth_dir,
                "Wan-AI",
                "Wan2.1-T2V-1.3B",
                "google",
                "umt5-xxl",
                "tokenizer.json",
            ),
        ]
        missing = [path for path in converted_files if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(
                "legacy FastWAM asset bundle is incomplete:\n  " + "\n  ".join(missing)
            )
        print("== legacy converted FastWAM components already cached ==")
    else:
        # The normal home path selects Wan's original public T5/VAE files. Match ModelConfig's
        # <base>/<repo_id>/<pattern> lookup exactly.
        component_specs = [
            (
                "Wan-AI/Wan2.2-TI2V-5B",
                "921dbaf3f1674a56f47e83fb80a34bac8a8f203e",
                ["models_t5_umt5-xxl-enc-bf16.pth", "Wan2.2_VAE.pth"],
            ),
            (
                "Wan-AI/Wan2.1-T2V-1.3B",
                "37ec512624d61f7aa208f7ea8140a131f93afc9a",
                ["google/umt5-xxl/**"],
            ),
        ]
        for repo_id, revision, patterns in component_specs:
            local_dir = os.path.join(diffsynth_dir, *repo_id.split("/"))
            if repo_id.endswith("Wan2.2-TI2V-5B"):
                ready = all(os.path.isfile(os.path.join(local_dir, name)) for name in patterns)
            else:
                ready = os.path.isfile(
                    os.path.join(local_dir, "google", "umt5-xxl", "tokenizer.json")
                )
            if ready:
                print(f"== FastWAM component already cached: {repo_id} ==")
                continue
            print(f"== downloading public FastWAM component: {repo_id} ==")
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                allow_patterns=patterns,
                local_dir=local_dir,
            )

    dataset_name = "libero_object_no_noops_lerobot"
    dataset_dir = os.path.join(data_dir, dataset_name)
    if include_dataset and not os.path.isfile(os.path.join(dataset_dir, "meta", "info.json")):
        print("== downloading public FastWAM LIBERO-Object replay data (~1.4 GB) ==")
        archive = snapshot_download(
            repo_id="yuanty/LIBERO-fastwam",
            repo_type="dataset",
            revision="ee018b997c430bb12b5bf3c892d744798c5a2f91",
            allow_patterns=[f"{dataset_name}.tar.gz"],
        )
        archive = os.path.join(archive, f"{dataset_name}.tar.gz")
        work = tempfile.mkdtemp(prefix=".fastwam-extract-", dir=data_dir)
        try:
            _safe_extract_tar(archive, work)
            extracted = os.path.join(work, dataset_name)
            if not os.path.isfile(os.path.join(extracted, "meta", "info.json")):
                raise RuntimeError(f"unexpected FastWAM dataset archive layout: {archive}")
            if os.path.exists(dataset_dir):
                shutil.rmtree(dataset_dir)
            shutil.move(extracted, dataset_dir)
        finally:
            shutil.rmtree(work, ignore_errors=True)
    elif include_dataset:
        print("== public FastWAM replay data already cached ==")

    return {
        "checkpoint": os.path.join(release_dir, release_files[0]),
        "dataset_stats": os.path.join(release_dir, release_files[1]),
        "dataset": dataset_dir if include_dataset else None,
    }


# ---------------------------------------------------------------------------------------
# DROID video decode + open-loop rollout
# ---------------------------------------------------------------------------------------
def _grab_frames(path, indices):
    """Decode the specific source frames we replan on (nearest-match fallback)."""
    import av

    want = sorted({int(i) for i in indices})
    if not want:
        return {}
    container = av.open(path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    rate, tb = float(stream.average_rate), stream.time_base
    try:
        container.seek(max(int((want[0] / rate) / tb), 0), stream=stream, backward=True, any_frame=False)
    except Exception:
        container.seek(0)
    out, remaining = {}, set(want)
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


def _decode_clip(path, start_idx, n_frames):
    """Decode n contiguous frames from start_idx for the episode video."""
    import av

    container = av.open(path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    rate, tb = float(stream.average_rate), stream.time_base
    try:
        container.seek(max(int((start_idx / rate) / tb), 0), stream=stream, backward=True, any_frame=False)
    except Exception:
        container.seek(0)
    frames = []
    for frame in container.decode(stream):
        if frame.pts is None:
            continue
        if int(round(float(frame.pts * tb) * rate)) < start_idx:
            continue
        frames.append(frame.to_ndarray(format="rgb24"))
        if len(frames) >= n_frames:
            break
    container.close()
    return frames


def _run_openloop_episode(base_policy, predict_chunk, norm_tag, num_steps, out_dir, exclude=None):
    """Replay one random DROID episode open-loop (replan every STRIDE) and show GT-vs-pred."""
    import matplotlib.pyplot as plt  # inline backend -> plots render in the notebook, no Agg/subprocess
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from IPython.display import Video, display
    from PIL import Image

    eval_repo = os.environ.get("EVAL_REPO", "allenai/MolmoAct2-DROID-Dataset")
    eval_revision = os.environ.get(
        "EVAL_REVISION",
        "e44d3138c64cfeb1c24fbbce087b475fb1233728",
    )
    stride = int(os.environ.get("STRIDE", "15"))
    view_cam = "observation.images." + os.environ.get("CAM", "exterior_1_left")
    cams = ["observation.images.exterior_1_left", "observation.images.exterior_2_left", "observation.images.wrist_left"]
    dim_names = ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]

    info_path = hf_hub_download(
        eval_repo,
        "meta/info.json",
        repo_type="dataset",
        revision=eval_revision,
    )
    info = json.load(open(info_path))
    fps, data_tmpl, video_tmpl = info["fps"], info["data_path"], info["video_path"]
    meta_ep = pq.read_table(
        hf_hub_download(
            eval_repo,
            "meta/episodes/chunk-000/file-000.parquet",
            repo_type="dataset",
            revision=eval_revision,
        )
    ).to_pydict()
    episodes = list(meta_ep["episode_index"])
    # Prefer already-cached episodes (including the old workshop subset). On a clean home setup,
    # download just one randomly selected episode's parquet + three camera files instead of the
    # complete DROID dataset.
    snap_root = os.path.dirname(os.path.dirname(info_path))
    _present = lambda rel: os.path.exists(os.path.join(snap_root, rel))

    def _episode_files(row):
        return [
            data_tmpl.format(
                chunk_index=meta_ep["data/chunk_index"][row],
                file_index=meta_ep["data/file_index"][row],
            ),
            *[
                video_tmpl.format(
                    video_key=c,
                    chunk_index=meta_ep[f"videos/{c}/chunk_index"][row],
                    file_index=meta_ep[f"videos/{c}/file_index"][row],
                )
                for c in cams
            ],
        ]

    exclude = set(exclude or ())
    available = [
        ep
        for r, ep in enumerate(episodes)
        if ep not in exclude and all(_present(rel) for rel in _episode_files(r))
    ]
    ep_env = os.environ.get("EPISODE", "")
    candidates = [ep for ep in episodes if ep not in exclude]
    episode = int(ep_env) if ep_env else random.choice(available or candidates)
    if episode not in episodes:
        raise ValueError(f"EPISODE={episode} is not present in {eval_repo}")
    mr = episodes.index(episode)
    if episode not in available:
        print(f"episode {episode} is not cached; downloading its DROID parquet + 3 camera files ...")
        try:
            for rel in _episode_files(mr):
                hf_hub_download(
                    eval_repo,
                    rel,
                    repo_type="dataset",
                    revision=eval_revision,
                )
        except Exception as exc:
            raise RuntimeError(
                f"Could not download DROID episode {episode} from {eval_repo}. "
                "Check network access/HF_TOKEN or choose a cached EPISODE."
            ) from exc
    d_chunk, d_file = meta_ep["data/chunk_index"][mr], meta_ep["data/file_index"][mr]
    length = meta_ep["length"][mr]
    task = meta_ep["tasks"][mr]
    task = task[0] if isinstance(task, (list, tuple)) else task
    print(f"episode {episode}  task={task!r}  length={length}")

    # dataset_from/to_index are GLOBAL rows; filter the per-file table by episode_index.
    dt = pq.read_table(
        hf_hub_download(
            eval_repo,
            data_tmpl.format(chunk_index=d_chunk, file_index=d_file),
            repo_type="dataset",
            revision=eval_revision,
        )
    )
    ep_table = dt.filter(pc.equal(dt.column("episode_index"), episode))
    col = lambda name: ep_table.column(name).to_pylist()
    states = np.asarray(col("observation.state"), np.float32)
    gt = np.asarray(col("action"), np.float32)
    timestamps = np.asarray(col("timestamp"), np.float64)

    replan_pts = list(range(0, length, stride))
    cam_meta = {
        c: {
            "chunk": meta_ep[f"videos/{c}/chunk_index"][mr],
            "file": meta_ep[f"videos/{c}/file_index"][mr],
            "from_ts": meta_ep[f"videos/{c}/from_timestamp"][mr],
        }
        for c in cams
    }
    frames_by_cam = {}
    for c in cams:
        cm = cam_meta[c]
        vp = hf_hub_download(
            eval_repo,
            video_tmpl.format(video_key=c, chunk_index=cm["chunk"], file_index=cm["file"]),
            repo_type="dataset",
            revision=eval_revision,
        )
        idxs = [int(round((cm["from_ts"] + timestamps[t]) * fps)) for t in replan_pts]
        grabbed = _grab_frames(vp, idxs)
        frames_by_cam[c] = {t: grabbed[int(round((cm["from_ts"] + timestamps[t]) * fps))] for t in replan_pts}

    pred = np.full_like(gt, np.nan)
    lat = []
    for t in replan_pts:
        pics = [Image.fromarray(frames_by_cam[c][t]) for c in cams]
        torch.cuda.synchronize()
        ts = time.perf_counter()
        chunk = predict_chunk(base_policy, norm_tag, pics, task, states[t], num_steps)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - ts) * 1000)
        n = min(stride, length - t, chunk.shape[0])
        pred[t : t + n] = chunk[:n]

    valid = ~np.isnan(pred).any(axis=1)
    l1 = float(np.abs(pred[valid] - gt[valid]).mean())
    mse = float(((pred[valid] - gt[valid]) ** 2).mean())
    print(f"open-loop        : L1={l1:.4f} MSE={mse:.4f}  ({np.mean(lat):.0f} ms/infer)")

    # Episode exterior view (the scene the model predicts on).
    cm = cam_meta[view_cam]
    vpath = hf_hub_download(
        eval_repo,
        video_tmpl.format(video_key=view_cam, chunk_index=cm["chunk"], file_index=cm["file"]),
        repo_type="dataset",
        revision=eval_revision,
    )
    frames = _decode_clip(vpath, int(round((cm["from_ts"] + timestamps[0]) * fps)), length)
    vid = os.path.join(out_dir, f"droid_ep{episode}.mp4")
    if frames:
        import imageio

        imageio.mimsave(vid, frames, fps=int(round(fps)), codec="libx264", quality=7)

    # GT (solid) vs predicted (dashed), overlaid per dim (workspace rule 2.a) - inline + saved.
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), squeeze=False)
    x = np.arange(gt.shape[0])
    for d in range(8):
        ax = axes[d // 4][d % 4]
        ax.plot(x, gt[:, d], color="tab:blue", lw=1.4, label="GT (teleop)")
        ax.plot(x, pred[:, d], color="tab:red", lw=1.2, ls="--", label="MolmoAct2 (pred)")
        ax.set_title(dim_names[d], fontsize=10)
        ax.tick_params(labelsize=7)
        if d == 0:
            ax.legend(fontsize=8, loc="best")
    fig.suptitle(
        f"MolmoAct2-DROID open-loop (ROCm), ep {episode}\n{task}\nL1={l1:.4f}  MSE={mse:.4f}  (GT solid, pred dashed)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(os.path.join(out_dir, f"droid_ep{episode}_actions.png"), dpi=110)
    plt.show()
    if frames:
        display(Video(vid, embed=True, width=480))
    return episode


def run_openloop_episodes(base_policy, predict_chunk, norm_tag, num_steps, out_dir):
    """Replay N_DROID_EPISODES random DROID episodes open-loop, rendering GT-vs-predicted actions
    (and the exterior-cam video) inline for each. `predict_chunk` is the notebook's flow-matching
    action-head call, passed in so the model-call semantics stay visible in the notebook."""
    n_droid = int(os.environ.get("N_DROID_EPISODES", "2"))
    seen = set()
    for _ in range(n_droid):
        seen.add(
            _run_openloop_episode(
                base_policy,
                predict_chunk,
                norm_tag,
                num_steps,
                out_dir,
                exclude=seen,
            )
        )
