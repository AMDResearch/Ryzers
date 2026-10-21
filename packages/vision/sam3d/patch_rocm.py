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
# Conditioning is fixed during each denoising stage. Keep conditional and CFG
# zero inputs separate, and release all cached tensors when the stage returns.
cache_import = ('from sam3d_objects.conditioning_cache import '
                'cache_sampling, conditioning_cast, conditioning_zero, conditioning_projection\n')
models = 'sam3d_objects/model/backbone/tdfy_dit/models/'
for name in ('mot_sparse_structure_flow.py', 'structured_latent_flow.py'):
    replace(models + name, 'import torch\n', cache_import + 'import torch\n')
    replace(models + name, 'cond = cond.type(self.dtype)',
            'cond = conditioning_cast(cond, self.dtype)')
    replace(models + name, 'cond = cond * 0', 'cond = conditioning_zero(cond)')
attention = 'sam3d_objects/model/backbone/tdfy_dit/modules/attention/modules.py'
replace(attention, 'import torch\n', cache_import + 'import torch\n')
replace(attention, 'kv = self.to_kv(context)',
        'kv = conditioning_projection(self.to_kv, context)')
replace(sparse + 'attention/modules.py', 'import torch\n', cache_import + 'import torch\n')
replace(sparse + 'attention/modules.py', 'kv = self._linear(self.to_kv, context)',
        'kv = conditioning_projection(self.to_kv, context, lambda: self._linear(self.to_kv, context))')
pipeline = 'sam3d_objects/pipeline/inference_pipeline.py'
replace(pipeline, 'import torch\n', cache_import + 'import torch\n')
for method in ('sample_sparse_structure', 'sample_slat'):
    replace(pipeline, f'    def {method}(', f'    @cache_sampling\n    def {method}(')
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
