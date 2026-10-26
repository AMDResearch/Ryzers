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
# Zero CFG context produces constant K/V. Keep one token before projection and
# normalization; the existing attention implementation handles the shorter input.
for path in (attention, sparse + 'attention/modules.py'):
    replace(path, 'import torch\n',
            'from sam3d_objects.conditioning_cache import compact_zero_context\nimport torch\n')
replace(attention, '            Lkv = context.shape[1]',
        '            context = compact_zero_context(self.to_kv, context, training=self.training)\n'
        '            Lkv = context.shape[1]')
replace(sparse + 'attention/modules.py', '            q = self._linear(self.to_q, x)',
        '            context = compact_zero_context(self.to_kv, context, training=self.training)\n'
        '            q = self._linear(self.to_q, x)')
# Guard normalization by stage, autocast mode and validated architecture.
norm_import = 'from sam3d_objects.model.backbone.tdfy_dit.modules import norm_rocm\n'
replace(attention, 'import torch\n', norm_import + 'import torch\n')
replace(attention,
        '        return (F.normalize(x, dim=-1) * self.gamma * self.scale).to(x.dtype)',
        '        if norm_rocm.can_use_dense_rms(self, x):\n'
        '            return norm_rocm.dense_rms(x, self.gamma, self.scale)\n'
        '        return (F.normalize(x, dim=-1) * self.gamma * self.scale).to(x.dtype)')
replace(sparse + 'attention/modules.py', 'import torch\n', norm_import + 'import torch\n')
replace(sparse + 'attention/modules.py', '        x_type = x.dtype',
        '        feats = x.feats if isinstance(x, SparseTensor) else x\n'
        '        if norm_rocm.can_use_rms(self, feats):\n'
        '            out = norm_rocm.sparse_rms(feats, self.gamma, self.scale)\n'
        '            return x.replace(out) if isinstance(x, SparseTensor) else out\n'
        '        x_type = x.dtype')
# Full attention already accepts separate Q/K/V. Avoid repacking normalized
# queries and keys alongside values into another allocation during inference.
replace(sparse + 'attention/modules.py',
        '                qkv = qkv.replace(torch.stack([q.feats, k.feats, v.feats], dim=1))',
        "                if self.attn_mode == 'full' and norm_rocm.can_use_separate_qkv(self.q_rms_norm, q.feats):\n"
        '                    h = sparse_scaled_dot_product_attention(q, k, v)\n'
        '                    h = self._reshape_chs(h, (-1,))\n'
        '                    return self._linear(self.to_out, h)\n'
        '                qkv = qkv.replace(torch.stack([q.feats, k.feats, v.feats], dim=1))')
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
# Keep the normalized FP16 latent MLP in BLASLt through its GELU epilogue.
sparse_blocks = sparse + 'transformer/blocks.py'
replace(sparse_blocks, 'import torch\n',
        'from sam3d_objects.linear_rocm import linear_gelu\nimport torch\n')
replace(sparse_blocks,
        '    def forward(self, x: SparseTensor) -> SparseTensor:\n'
        '        return self.mlp(x)',
        '    def forward(self, x: SparseTensor) -> SparseTensor:\n'
        '        if (type(self.mlp) is nn.Sequential and len(self.mlp) == 3\n'
        '                and type(self.mlp[0]) is SparseLinear\n'
        '                and type(self.mlp[1]) is SparseGELU\n'
        '                and type(self.mlp[2]) is SparseLinear\n'
        '                and self.mlp[1].approximate == "tanh"\n'
        '                and not any(m._forward_hooks or m._forward_pre_hooks\n'
        '                            for m in (self.mlp, self.mlp[1]))):\n'
        '            hidden = linear_gelu(self.mlp[0], x.feats)\n'
        '            if hidden is not None:\n'
        '                return self.mlp[2](x.replace(hidden))\n'
        '        return self.mlp(x)')
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
# Select convolutions before the first depth inference. Timing-based solver
# selection on gfx1151 can change FP16 depth values between fresh processes.
# Preserve explicit runtime settings and the existing defaults on other GPUs.
replace(pipeline, '        self.device = torch.device(device)',
        '        self.device = torch.device(device)\n'
        "        if (torch.version.hip and self.device.type == 'cuda' and\n"
        "                torch.cuda.get_device_properties(self.device).gcnArchName.split(':')[0] == 'gfx1151'):\n"
        "            os.environ.setdefault('MIOPEN_FIND_MODE', 'FAST')")
replace(pipeline, 'import torch\n', cache_import + 'import torch\n')
replace(pipeline, 'import torch\n',
        'from sam3d_objects.backbone_rocm import prepare_inference_memory\nimport torch\n')
replace(pipeline, '            logger.info("Loading model weights completed!")',
        '            prepare_inference_memory(self)\n'
        '            logger.info("Loading model weights completed!")')
for path in (pipeline, 'sam3d_objects/pipeline/inference_pipeline_pointmap.py'):
    replace(path, 'import torch\n',
            'from sam3d_objects.backbone_rocm import cache_image_features\nimport torch\n')
    replace(path, '    def run(', '    @cache_image_features\n    def run(')
dino = 'sam3d_objects/model/backbone/dit/embedder/dino.py'
replace(dino, 'import torch\n',
        'from sam3d_objects.backbone_rocm import backbone_features\nimport torch\n')
replace(dino, '        output = self.backbone.forward_features(input_img)',
        '        output = backbone_features(self.backbone, input_img)')
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
