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
# Restrict fused normalization to sparse-latent inference.
norm_import = 'from sam3d_objects.model.backbone.tdfy_dit.modules import norm_rocm\n'
replace(sparse + 'attention/modules.py', 'import torch\n', norm_import + 'import torch\n')
replace(sparse + 'attention/modules.py', '        x_type = x.dtype',
        '        feats = x.feats if isinstance(x, SparseTensor) else x\n'
        '        if norm_rocm.can_use_rms(self, feats):\n'
        '            out = norm_rocm.sparse_rms(feats, self.gamma, self.scale)\n'
        '            return x.replace(out) if isinstance(x, SparseTensor) else out\n'
        '        x_type = x.dtype')
norm = 'sam3d_objects/model/backbone/tdfy_dit/modules/norm.py'
replace(norm, 'import torch\n', norm_import + 'import torch\n')
replace(norm, 'class LayerNorm32(nn.LayerNorm):\n'
        '    def forward(self, x: torch.Tensor) -> torch.Tensor:\n',
        'class LayerNorm32(nn.LayerNorm):\n'
        '    def forward(self, x: torch.Tensor) -> torch.Tensor:\n'
        '        if type(self) is LayerNorm32 and norm_rocm.can_use_layer(self, x):\n'
        '            return norm_rocm.layer_norm(x, self.weight, self.bias, self.eps)\n')
# Construct normal Linear subclasses, preserving parameter/state-dict names.
# Their cast cache is active only during sparse-structure inference.
for path in (attention, 'sam3d_objects/model/backbone/tdfy_dit/modules/transformer/blocks.py',
             'sam3d_objects/model/backbone/tdfy_dit/modules/transformer/modulated.py'):
    replace(path, 'import torch\n',
            'from sam3d_objects.linear_rocm import InferenceLinear\nimport torch\n')
    replace(path, 'nn.Linear(', 'InferenceLinear(')
# Preserve SparseLinear's feature wrapper and parameter names.
replace(sparse + 'linear.py', 'import torch\n',
        'from sam3d_objects.linear_rocm import InferenceLinear\nimport torch\n')
replace(sparse + 'linear.py', 'class SparseLinear(nn.Linear):',
        'class SparseLinear(InferenceLinear):')
# Share the packed float32 attribute buffer with the PLY writer. Upstream's
# per-row Python tuples duplicate gigabytes of RAM used by the iGPU as well.
gaussian = 'sam3d_objects/model/backbone/tdfy_dit/representations/gaussian/gaussian_model.py'
replace(gaussian, '        elements = np.empty(xyz.shape[0], dtype=dtype_full)\n', '')
replace(gaussian, '        elements[:] = list(map(tuple, attributes))',
        '        elements = np.ascontiguousarray(attributes, dtype=np.float32).view(\n'
        '            np.dtype(dtype_full)\n'
        '        ).reshape(-1)')
# Fuse only the qualified elementwise operations, preserving FP16 intermediates.
modulated = 'sam3d_objects/model/backbone/tdfy_dit/modules/transformer/modulated.py'
replace(modulated, 'import torch\n',
        'from sam3d_objects.model.backbone.tdfy_dit.modules import pointwise_rocm\nimport torch\n')
for name in ('msa', 'mlp'):
    replace(modulated,
            f'        return h * (1 + scale_{name}.unsqueeze(1)) + shift_{name}.unsqueeze(1)',
            f'        return pointwise_rocm.modulate(h, scale_{name}, shift_{name}, training=self.training)')
replace(modulated, '    def _apply_add(self, x, h):',
        '    def _apply_gated_add(self, x, h, multiplier):\n'
        '        return pointwise_rocm.gated_add(x, h, multiplier, training=self.training)\n\n'
        '    def _apply_add(self, x, h):')
for gate in ('gate_msa', 'gate_mlp'):
    replace(modulated,
            '        h = _pytree.tree_map(\n'
            f'            partial(self._apply_multiplication, multiplier={gate}),\n'
            '            h\n'
            '        )\n'
            '        x = _pytree.tree_map(\n'
            '            self._apply_add,\n'
            '            x,\n'
            '            h\n'
            '        )',
            f'        x = _pytree.tree_map(partial(self._apply_gated_add, multiplier={gate}), x, h)')
pipeline = 'sam3d_objects/pipeline/inference_pipeline.py'
replace(pipeline, 'import torch\n', cache_import + 'import torch\n')
# All 256 uint8 quotients round identically without a full float64 image.
replace(pipeline, '        image = np.array(image)\n'
        '        image = image / 255\n'
        '        image = image.astype(np.float32)\n'
        '        return image',
        '        image = np.asarray(image)\n'
        '        if image.dtype == np.uint8:\n'
        '            return np.divide(image, 255, dtype=np.float32)\n'
        '        return (image / 255).astype(np.float32)')
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
