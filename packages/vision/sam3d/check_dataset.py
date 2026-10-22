# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""CPU mesh-quality metrics against independent SA-3DAO artist references.

Definitions follow paper v2 D.3.1. Reports record normalization, alignment,
sampling and distance conventions for reproducible comparisons.
"""

import json, time, hashlib, itertools
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree

PROTOCOL = {
    "reference": "SA-3DAO public artist-created GLB meshes",
    "paper": "https://arxiv.org/html/2511.16624v2#A4.SS3.SSS1",
    "normalization": "Independent bbox-center translation and uniform scaling by 2 / "
    "longest bbox extent; preserves aspect ratios.",
    "alignment": "Rigid proper rotations/translations only, no reflections or fitted "
    "scaling; deterministic multi-start PCA-axis ICP (24 proper hypotheses "
    "plus identity), 2048 coarse source points / 8192 target points, then "
    "best three hypotheses refined with 20000 / 50000 points.",
    "surface_samples": 1000000,
    "f_score_threshold": 0.01,
    "chamfer": "Report mean bidirectional unsquared distance, sum of directional means, "
    "and sum of directional mean squared distances explicitly. Paper does not specify "
    "all implementation conventions.",
    "voxel_iou": "Voxelize surface samples on one common isotropic 64^3 grid enclosing "
    "both aligned point sets. Occupancy IoU, not filled-solid IoU.",
    "scope": "Local implementation of published metric definitions with explicitly "
    "documented conventions; not an official leaderboard submission. Compare "
    "scores only on matching splits and inference settings.",
}


def load_mesh(path):
    obj = trimesh.load(str(path), force="scene", process=False)
    mesh = obj.to_mesh() if isinstance(obj, trimesh.Scene) else obj
    assert (
        len(mesh.vertices) > 0
        and len(mesh.faces) > 0
        and np.isfinite(mesh.vertices).all()
    ), str(path)
    return mesh


def normalize(mesh):
    mesh = mesh.copy()
    bounds = np.asarray(mesh.bounds)
    center = bounds.mean(0)
    extent = (bounds[1] - bounds[0]).max()
    assert extent > 0
    mesh.vertices = (np.asarray(mesh.vertices) - center) * (2 / extent)
    return mesh, {"center": center.tolist(), "scale": float(2 / extent)}


def sample(mesh, n, seed):
    return np.ascontiguousarray(
        trimesh.sample.sample_surface(mesh, n, seed=seed)[0], dtype=np.float64
    )


def icp(source, target, R=None, t=None, iterations=30, threshold=0.3):
    R = np.eye(3) if R is None else R.copy()
    t = np.zeros(3) if t is None else t.copy()
    tree = cKDTree(target)
    last = np.inf
    for _ in range(iterations):
        moved = source @ R.T + t
        dist, idx = tree.query(moved, workers=1)
        keep = dist < threshold
        if keep.sum() < 12:
            break
        a = moved[keep]
        b = target[idx[keep]]
        ca = a.mean(0)
        cb = b.mean(0)
        u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
        delta = vt.T @ u.T
        if np.linalg.det(delta) < 0:
            vt[-1] *= -1
            delta = vt.T @ u.T
        dt = cb - delta @ ca
        R = delta @ R
        t = delta @ t + dt
        cost = np.mean(dist[keep] ** 2)
        if abs(last - cost) < 1e-10:
            break
        last = cost
    return R, t


def align(source, target):
    a = source[:2048]
    b = target[:8192]
    ca = a.mean(0)
    cb = b.mean(0)
    _, _, va = np.linalg.svd(a - ca, full_matrices=False)
    _, _, vb = np.linalg.svd(b - cb, full_matrices=False)
    starts = [np.eye(3)]
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([-1.0, 1.0], repeat=3):
            p = np.eye(3)[list(perm)] * np.array(signs)[:, None]
            r = vb.T @ p @ va
            if np.linalg.det(r) > 0.999:
                starts.append(r)
    tree_b = cKDTree(b)
    candidates = []
    for initial in starts:
        R, t = icp(a, b, initial, cb - initial @ ca)
        moved = a @ R.T + t
        score = (
            tree_b.query(moved, workers=1)[0].mean()
            + cKDTree(moved).query(b[:2048], workers=1)[0].mean()
        )
        candidates.append((float(score), R, t))
    a = source[:20000]
    b = target[:50000]
    tree_b = cKDTree(b)
    fine = []
    for _, R, t in sorted(candidates, key=lambda x: x[0])[:3]:
        R, t = icp(a, b, R, t, iterations=60, threshold=0.2)
        moved = a @ R.T + t
        score = (
            tree_b.query(moved, workers=1)[0].mean()
            + cKDTree(moved).query(b[:20000], workers=1)[0].mean()
        )
        fine.append((float(score), R, t))
    return min(fine, key=lambda x: x[0])[1:]


def metrics(a, b, workers=1):
    da = cKDTree(b).query(a, workers=workers)[0]
    db = cKDTree(a).query(b, workers=workers)[0]
    precision = float(np.mean(da < 0.01))
    recall = float(np.mean(db < 0.01))
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    low = np.minimum(a.min(0), b.min(0))
    high = np.maximum(a.max(0), b.max(0))
    width = (high - low).max()
    low = (low + high) / 2 - width / 2

    def voxels(x):
        q = np.clip(np.floor((x - low) * (64 / width)).astype(np.int32), 0, 63)
        return np.unique(q[:, 0] * 4096 + q[:, 1] * 64 + q[:, 2])

    av = voxels(a)
    bv = voxels(b)
    inter = np.intersect1d(av, bv, assume_unique=True).size
    return {
        "f_score_001": f1,
        "precision_001": precision,
        "recall_001": recall,
        "chamfer_l1_mean": float((da.mean() + db.mean()) / 2),
        "chamfer_l1_sum": float(da.mean() + db.mean()),
        "chamfer_squared_sum": float(np.mean(da**2) + np.mean(db**2)),
        "voxel_iou_64": float(inter / (len(av) + len(bv) - inter)),
        "predicted_occupied_voxels": len(av),
        "reference_occupied_voxels": len(bv),
    }


def self_check():
    rng = np.random.default_rng(42)
    a = rng.normal(size=(4096, 3))
    a /= np.linalg.norm(a, axis=1)[:, None]
    a *= np.array([1, 0.7, 0.4])
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    b = a @ q.T + np.array([0.12, -0.09, 0.17])
    R, t = align(a, b)
    m = metrics(a @ R.T + t, b)
    assert m["f_score_001"] == 1 and m["chamfer_l1_mean"] < 1e-8, m
    identity = metrics(a, a)
    assert identity["voxel_iou_64"] == 1 and identity["chamfer_l1_sum"] == 0
    negative = metrics(a, a * np.array([1, 0.15, 1]))
    assert negative["f_score_001"] < 0.2 and negative["voxel_iou_64"] < 0.2, negative
    return {
        "known_rigid_transform": m,
        "identity": identity,
        "deformation_negative_control": negative,
    }


def reconstruct(references, predictions):
    """Generate meshes with the production RGB-only API before CPU scoring."""
    import gc
    import sys

    import torch
    from huggingface_hub import snapshot_download
    from PIL import Image

    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError(
            "--reconstruct requires an accessible ROCm GPU; omit it to score existing meshes on CPU"
        )

    sys.path.insert(0, "/ryzers/sam-3d-objects/notebook")
    from inference import Inference

    if any((predictions / p.name / "mesh.glb").exists() for p in references):
        raise FileExistsError("Use an empty prediction directory for --reconstruct")
    revision = "2e73555018d2741ccd486e56c24fac41155a1dc6"
    checkpoint = (
        Path(
            snapshot_download(
                "facebook/sam-3d-objects",
                revision=revision,
                allow_patterns=["checkpoints/*"],
            )
        )
        / "checkpoints"
    )
    torch.manual_seed(42)
    inference = Inference(str(checkpoint / "pipeline.yaml"), compile=False)
    metadata = {
        "model_revision": revision,
        "seed": 42,
        "compile": False,
        "steps": "defaults",
        "decoders": "mesh and gaussian",
        "pointmap": "estimated by production MoGe path from RGB",
        "mask": "grayscale JPEG >127",
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "rocm": torch.version.hip,
        "cpu_threads": torch.get_num_threads(),
    }
    predictions.mkdir(parents=True, exist_ok=True)
    (predictions / "inference-settings.json").write_text(json.dumps(metadata, indent=2))
    for folder in references:
        destination = predictions / folder.name
        destination.mkdir(parents=True, exist_ok=True)
        row = {"object_id": folder.name}
        output = None
        try:
            image = np.asarray(Image.open(folder / "image.jpg").convert("RGB"))
            mask = np.asarray(Image.open(folder / "mask.jpg").convert("L")) > 127
            if mask.shape != image.shape[:2] or not mask.any():
                raise ValueError(
                    "Image and nonempty mask must have matching dimensions"
                )
            output = inference(image, mask, seed=42)
            mesh = output["glb"]
            if (
                mesh is None
                or not len(mesh.faces)
                or not np.isfinite(mesh.vertices).all()
            ):
                raise ValueError("Reconstruction produced invalid mesh geometry")
            if not torch.isfinite(output["gaussian"][0].get_xyz).all().item():
                raise ValueError("Reconstruction produced nonfinite Gaussian centers")
            mesh.export(str(destination / "mesh.glb"))
            row.update(status="ok", vertices=len(mesh.vertices), faces=len(mesh.faces))
        except Exception as error:
            row.update(
                status="error", error_type=type(error).__name__, error=str(error)
            )
        finally:
            output = None
            (destination / "inference.json").write_text(json.dumps(row, indent=2))
            print(json.dumps(row), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    del inference
    gc.collect()
    torch.cuda.empty_cache()
    return metadata


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, help="Extracted SA-3DAO reference dataset"
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Prediction directory containing <object_id>/mesh.glb",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("/ryzers/output/sa3dao-scores.json")
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000000,
        help="Surface points per mesh; published protocol uses 1M",
    )
    parser.add_argument(
        "--nn-workers",
        type=int,
        default=2,
        help="CPU threads for exact nearest-neighbor queries",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Check identity, known rigid motion and deformation without data or GPU",
    )
    parser.add_argument(
        "--reconstruct",
        action="store_true",
        help="Run production inference into an empty prediction directory before scoring",
    )
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_check(), indent=2))
        return
    if args.dataset_root is None or args.predictions is None:
        parser.error(
            "--dataset-root and --predictions are required unless --self-test is selected"
        )
    if args.samples < 2048 or args.nn_workers < 1:
        parser.error("--samples must be >=2048 and --nn-workers must be positive")
    references = sorted(
        p for p in args.dataset_root.glob("sa3dao_*") if (p / "mesh.glb").is_file()
    )
    if not references:
        parser.error("No SA-3DAO object folders with reference mesh.glb files found")
    protocol = dict(
        PROTOCOL, surface_samples=args.samples, nearest_neighbor_workers=args.nn_workers
    )
    report = {"protocol": protocol, "reference_count": len(references), "objects": []}
    if args.reconstruct:
        report["inference"] = reconstruct(references, args.predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for folder in references:
        started = time.monotonic()
        row = {"object_id": folder.name}
        try:
            predicted, pnorm = normalize(
                load_mesh(args.predictions / folder.name / "mesh.glb")
            )
            reference, rnorm = normalize(load_mesh(folder / "mesh.glb"))
            seed = int.from_bytes(
                hashlib.sha256(folder.name.encode()).digest()[:4], "little"
            )
            a = sample(predicted, args.samples, seed)
            b = sample(reference, args.samples, seed ^ 0x91F08B3C)
            rotation, translation = align(a, b)
            row.update(
                status="ok",
                **metrics(a @ rotation.T + translation, b, workers=args.nn_workers),
                pred_normalization=pnorm,
                reference_normalization=rnorm,
                alignment_rotation=rotation.tolist(),
                alignment_translation=translation.tolist(),
            )
        except Exception as error:
            row.update(
                status="error", error_type=type(error).__name__, error=str(error)
            )
        row["seconds"] = time.monotonic() - started
        report["objects"].append(row)
        good = [r for r in report["objects"] if r["status"] == "ok"]
        report["completed"] = len(report["objects"])
        report["successful"] = len(good)
        report["means"] = (
            {
                key: float(np.mean([r[key] for r in good]))
                for key in [
                    "f_score_001",
                    "voxel_iou_64",
                    "chamfer_l1_mean",
                    "chamfer_l1_sum",
                    "chamfer_squared_sum",
                ]
            }
            if good
            else {}
        )
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output)
        print(json.dumps(row), flush=True)
    if report["successful"] != len(references):
        raise SystemExit(
            "Some reference/prediction pairs could not be scored; see the report"
        )


if __name__ == "__main__":
    main()
