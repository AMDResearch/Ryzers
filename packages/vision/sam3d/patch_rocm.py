#!/usr/bin/env python3
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Small, checked adaptations for the pinned upstream SAM 3D source."""
from pathlib import Path

ROOT = Path('/ryzers/sam-3d-objects')


def replace(path, old, new):
    source = ROOT / path
    text = source.read_text()
    if old not in text:
        raise RuntimeError(f'Upstream patch context missing: {path}: {old!r}')
    source.write_text(text.replace(old, new))


sparse = 'sam3d_objects/model/backbone/tdfy_dit/modules/sparse/'
replace(sparse + 'basic.py', 'from spconv.pytorch import', 'from spconv_triton.pytorch import')
replace(sparse + 'conv/conv_spconv.py', 'import spconv.pytorch as spconv', 'import spconv_triton.pytorch as spconv')
# Keep upstream's masked implementation as a numerical reference, and route its
# callers through independent unmasked blocks to avoid quadratic mask storage.
replace(sparse + 'attention/masked_sdpa.py', 'import torch.nn.functional as F',
        'import torch.nn.functional as F\nfrom .sdpa_rocm import masked_sdpa')
replace(sparse + 'attention/masked_sdpa.py',
        'def masked_sdpa(q, k, v, q_seqlen, kv_seqlen):',
        'def reference_masked_sdpa(q, k, v, q_seqlen, kv_seqlen):')
# These Kaolin notebook imports are unused; reconstruction does not need them.
replace('notebook/inference.py', 'from kaolin.visualize import IpyTurntableVisualizer\n', '')
replace('notebook/inference.py', 'from kaolin.render.camera import Camera, CameraExtrinsics, PinholeIntrinsics\n', '')
# FlexiCubes itself is pure torch. Replace its only Kaolin utility dependency
# with the shape check used at its call sites, preserving input validation.
replace('sam3d_objects/model/backbone/tdfy_dit/representations/mesh/flexicubes/flexicubes.py',
        'from kaolin.utils.testing import check_tensor',
        '''def check_tensor(tensor, shape, throw=True):
    valid = isinstance(tensor, torch.Tensor) and tensor.ndim == len(shape)
    valid = valid and all(s is None or s == d for s, d in zip(shape, tensor.shape))
    if throw and not valid:
        raise ValueError(f"Expected tensor of shape {shape}")
    return valid''')
