#!/usr/bin/env python3
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Run the upstream SAM 3D Objects kidsroom example on ROCm."""
import argparse
import json
from pathlib import Path
import sys
import time


def check_sparse_convolution():
    """Compare the ROCm sparse backend against a dense torch reference."""
    import torch
    from sam3d_objects.model.backbone.tdfy_dit.modules.sparse import SparseTensor, SparseConv3d
    torch.manual_seed(42)
    coords = torch.cartesian_prod(torch.arange(2), *(torch.arange(5) for _ in range(3)))
    coords = coords[torch.rand(len(coords)) > 0.35].to(device='cuda', dtype=torch.int32)
    features = torch.randn(len(coords), 8, device='cuda')
    x = SparseTensor(feats=features, coords=coords)
    conv = SparseConv3d(8, 8, 3, bias=True).cuda().eval()
    with torch.no_grad():
        actual = conv(x)
        dense = x.dense()
        weights = conv.conv.weight.permute(0, 4, 1, 2, 3).contiguous()
        reference = torch.nn.functional.conv3d(dense, weights, conv.conv.bias, padding=1)
        b, z, y, xcoord = actual.coords.long().unbind(1)
        expected = reference[b, :, z, y, xcoord]
        torch.testing.assert_close(actual.feats, expected, rtol=2e-4, atol=2e-4)
    print('Sparse convolution matches dense PyTorch reference')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true', help='check GPU and imports without downloading weights')
    parser.add_argument('--output', type=Path, default=Path('/ryzers/output'))
    masks = parser.add_mutually_exclusive_group()
    masks.add_argument('--mask-index', type=int, nargs='+', default=[14])
    masks.add_argument('--all-masks', action='store_true', help='run the upstream multi-object notebook masks')
    args = parser.parse_args()
    import torch
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError('SAM 3D requires an accessible ROCm GPU')
    print(f'PyTorch {torch.__version__}; ROCm {torch.version.hip}; GPU {torch.cuda.get_device_name(0)}')
    root = Path('/ryzers/sam-3d-objects')
    sys.path.insert(0, str(root / 'notebook'))
    from inference import Inference, load_image, load_single_mask
    check_sparse_convolution()
    if args.smoke:
        print('SAM 3D inference imports passed')
        return
    from huggingface_hub import snapshot_download
    checkpoint_root = Path(snapshot_download('facebook/sam-3d-objects', allow_patterns=['checkpoints/*'])) / 'checkpoints'
    example = root / 'notebook/images/shutterstock_stylish_kidsroom_1640806567'
    image = load_image(str(example / 'image.png'))
    indices = sorted(int(p.stem) for p in example.glob('[0-9]*.png')) if args.all_masks else args.mask_index
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    inference = Inference(str(checkpoint_root / 'pipeline.yaml'), compile=False)
    load_seconds = round(time.monotonic() - start, 1)
    outputs = []
    for index in indices:
        torch.cuda.reset_peak_memory_stats()
        run_start = time.monotonic()
        mask = load_single_mask(str(example), index=index)
        assert image.shape[:2] == mask.shape and mask.any(), 'Invalid example image/mask'
        output = inference(image, mask, seed=42)
        report = save_output(output, args.output / f'kidsroom_{index}')
        report.update({
            'example': str(example.relative_to(root)), 'mask_index': index,
            'model_revision': checkpoint_root.parent.name,
            'model_load_seconds': load_seconds,
            'seed': 42, 'torch': torch.__version__, 'rocm': torch.version.hip,
            'gpu': torch.cuda.get_device_name(0),
            'seconds': round(time.monotonic() - run_start, 1),
            'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
        })
        (args.output / f'kidsroom_{index}.json').write_text(json.dumps(report, indent=2) + '\n')
        print('PASS:', json.dumps(report))
        # Keep only what upstream make_scene needs, releasing mesh intermediates.
        outputs.append({k: output[k] for k in ('gaussian', 'rotation', 'translation', 'scale')})
        del output
    if len(outputs) > 1:
        from inference import make_scene
        scene = make_scene(*outputs)
        scene_path = args.output / 'kidsroom_scene.ply'
        scene.save_ply(str(scene_path))
        count = check_ply(scene_path)
        assert count == sum(len(out['gaussian'][0].get_xyz) for out in outputs)
        print(f'PASS: assembled {len(outputs)} objects, {count} Gaussians; {time.monotonic()-start:.1f}s total')


def check_ply(path):
    from plyfile import PlyData
    import numpy as np
    data = PlyData.read(str(path))['vertex']
    xyz = np.column_stack([data[k] for k in ('x', 'y', 'z')])
    assert len(xyz) > 0 and np.isfinite(xyz).all(), 'Invalid reconstructed splat'
    from scipy.special import expit
    for field in data.data.dtype.names:
        # Upstream stores opacity as logits: +/-inf legitimately encodes 1/0.
        values = expit(data[field]) if field == 'opacity' else data[field]
        assert np.isfinite(values).all(), f'Invalid PLY field: {field}'
    return len(xyz)


def save_output(output, stem):
    import numpy as np
    import trimesh
    ply = stem.with_suffix('.ply')
    output['gs'].save_ply(str(ply))
    count = check_ply(ply)
    mesh = output['glb']
    assert mesh is not None and len(mesh.vertices) and len(mesh.faces), 'Empty mesh'
    assert np.isfinite(mesh.vertices).all(), 'Nonfinite mesh vertices'
    glb = stem.with_suffix('.glb')
    mesh.export(str(glb))
    loaded = trimesh.load(str(glb), force='mesh')
    assert len(loaded.vertices) and len(loaded.faces), 'Invalid exported GLB'
    assert np.isfinite(loaded.vertices).all(), 'Nonfinite exported GLB'
    return {'gaussians': count, 'vertices': len(loaded.vertices), 'faces': len(loaded.faces)}


if __name__ == '__main__':
    main()
