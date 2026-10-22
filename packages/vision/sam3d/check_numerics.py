#!/usr/bin/env python3
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Check ROCm kernels, image/PLY conversion and frozen CUDA sparse fixtures."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys


def check_attention():
    import torch
    module = importlib.import_module(
        'sam3d_objects.model.backbone.tdfy_dit.modules.sparse.attention.masked_sdpa')
    results = []
    torch.manual_seed(123)
    # Include unequal lengths and multiple blocks, so a missing block mask fails.
    for dtype, atol in [(torch.float32, 2e-5), (torch.float16, 2e-3), (torch.bfloat16, 2e-2)]:
        for qlens, klens in [([17], [31]), ([1024], [1024]), ([13, 0, 19], [7, 5, 29])]:
            q = torch.randn(1, sum(qlens), 2, 64, device='cuda', dtype=dtype)
            k = torch.randn(1, sum(klens), 2, 64, device='cuda', dtype=dtype)
            v = torch.randn_like(k)
            actual = module.masked_sdpa(q, k, v, qlens, klens).double().cpu()
            # Independent CPU float64 softmax(QK^T / sqrt(d)) V, block by block.
            references = []
            qi = ki = 0
            for qlen, klen in zip(qlens, klens):
                qc = q[0, qi:qi+qlen].double().cpu().transpose(0, 1)
                kc = k[0, ki:ki+klen].double().cpu().permute(1, 2, 0)
                vc = v[0, ki:ki+klen].double().cpu().transpose(0, 1)
                references.append((((qc @ kc) / 8).softmax(-1) @ vc).transpose(0, 1))
                qi += qlen
                ki += klen
            expected = torch.cat(references)
            torch.testing.assert_close(actual, expected, atol=atol, rtol=0)
            error = actual - expected
            results.append({'dtype': str(dtype), 'q_lengths': qlens, 'kv_lengths': klens,
                            'max_abs': error.abs().max().item(),
                            'rms': error.square().mean().sqrt().item(), 'atol': atol})
    return results


def check_triton_attention():
    import torch
    from sam3d_objects.conditioning_cache import sampling_cache
    from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.attention.attention_rocm import can_use_attention
    from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.attention.masked_sdpa import masked_sdpa
    torch.manual_seed(123)
    results = []
    # Include tails and packed Q/K/V strides. Sample queries against the entire
    # key/value context in CPU float64, without materializing a full CPU mask.
    for nq, nk in [(2049, 2071), (4096, 4096)]:
        q = torch.randn(1, nq, 3, 16, 64, device='cuda', dtype=torch.float16)[:, :, 0]
        kv = torch.randn(1, nk, 2, 16, 64, device='cuda', dtype=torch.float16)
        k, v = kv.unbind(2)
        indices = torch.linspace(0, nq-1, 32).long()
        with torch.no_grad(), sampling_cache('sample_slat'):
            backend = 'triton' if can_use_attention(q[0], k[0], v[0]) else 'torch'
            actual = masked_sdpa(q, k, v, [nq], [nk]).cpu()[indices].double()
        qc = q[0].cpu()[indices].double().transpose(0, 1)
        kc = k[0].cpu().double().permute(1, 2, 0)
        vc = v[0].cpu().double().transpose(0, 1)
        expected = (((qc @ kc) / 8).softmax(-1) @ vc).transpose(0, 1)
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=0)
        with torch.no_grad(), sampling_cache('sample_sparse_structure'):
            assert not can_use_attention(q[0], k[0], v[0]), 'Structure sampling must retain upstream SDPA'
        results.append({'q_length': nq, 'kv_length': nk, 'backend': backend,
                        'sampled_queries': len(indices),
                        'max_abs': (actual-expected).abs().max().item(), 'atol': 2e-3})
    return results


def check_normalization():
    import torch
    from sam3d_objects.conditioning_cache import sampling_cache
    from sam3d_objects.model.backbone.tdfy_dit.modules import norm_rocm
    from sam3d_objects.model.backbone.tdfy_dit.modules.attention.modules import MultiHeadRMSNorm
    from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.attention.modules import SparseMultiHeadRMSNorm
    from sam3d_objects.model.backbone.tdfy_dit.modules.norm import LayerNorm32
    torch.manual_seed(123)
    results = []
    module = SparseMultiHeadRMSNorm(64, 16).cuda().eval()
    with torch.no_grad():
        module.gamma.copy_(torch.randn_like(module.gamma) * 0.1 + 1)
    x = torch.randn(2, 513, 3, 16, 64, device='cuda', dtype=torch.float16)[:, :, 0]
    for autocast in (False, True):
        with torch.no_grad(), sampling_cache('sample_slat'), torch.autocast('cuda', enabled=autocast):
            actual = module(x).cpu().double()
            backend = 'triton' if norm_rocm.can_use_rms(module, x) else 'torch'
        xc = x.cpu().double()
        expected = xc / xc.square().sum(-1, keepdim=True).sqrt().clamp_min(1e-12)
        expected = expected * module.gamma.detach().cpu().double() * module.scale
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
        results.append({'kind': 'sparse_rms', 'autocast': autocast, 'backend': backend,
                        'max_abs': (actual-expected).abs().max().item(), 'atol': 2e-3, 'rtol': 2e-3})
    with torch.no_grad(), sampling_cache('sample_sparse_structure'):
        assert not norm_rocm.can_use_rms(module, x)
    with sampling_cache('sample_slat'):
        assert not norm_rocm.can_use_rms(module, x), 'Gradients require PyTorch'
    module = MultiHeadRMSNorm(64, 16).cuda().eval()
    with torch.no_grad():
        module.gamma.copy_(torch.randn_like(module.gamma) * 0.1 + 1)
    for tokens in (1, 4, 33, 4096):
        for magnitude in (0., 1., 2**-14, 2**-24):
            x = torch.randn(1, tokens, 3, 16, 64, device='cuda', dtype=torch.float16)[:, :, 0]
            x.mul_(magnitude)
            with torch.no_grad(), sampling_cache('sample_sparse_structure'), torch.autocast('cuda'):
                actual = module(x)
                expected = (torch.nn.functional.normalize(x, dim=-1)
                            * module.gamma * module.scale).to(x.dtype)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                backend = 'triton' if norm_rocm.can_use_dense_rms(module, x) else 'torch'
                results.append({'kind': 'dense_rms', 'tokens': tokens, 'magnitude': magnitude,
                                'backend': backend, 'max_abs': (actual-expected).abs().max().item()})
    with torch.no_grad(), sampling_cache('sample_sparse_structure'):
        assert not norm_rocm.can_use_dense_rms(module, x), 'Dense fusion requires FP16 autocast'
    for channels in (1024, 2048):
        module = LayerNorm32(channels, eps=1e-6).cuda().eval()
        for offset in (0., 16.):
            x = (torch.randn(513, channels, device='cuda') * 0.03125 + offset).half()
            with torch.no_grad(), sampling_cache('sample_slat'):
                actual = module(x).cpu().double()
                backend = 'triton' if norm_rocm.can_use_layer(module, x) else 'torch'
            xc = x.cpu().double()
            centered = xc - xc.mean(-1, keepdim=True)
            expected = centered / (centered.square().mean(-1, keepdim=True) + module.eps).sqrt()
            expected = expected * module.weight.detach().cpu().double() + module.bias.detach().cpu().double()
            torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
            results.append({'kind': 'layer_norm', 'channels': channels, 'offset': offset,
                            'backend': backend, 'max_abs': (actual-expected).abs().max().item(),
                            'atol': 2e-3, 'rtol': 2e-3})
    return results


def check_linear_cache():
    import torch
    import torch.nn.functional as F
    from sam3d_objects.conditioning_cache import sampling_cache, _weights
    from sam3d_objects.linear_rocm import InferenceLinear
    torch.manual_seed(123)
    module = InferenceLinear(1024, 2048).cuda().eval().requires_grad_(False)
    x = torch.randn(33, 1024, device='cuda')
    maximum = 0.
    with torch.no_grad(), sampling_cache('sample_sparse_structure'), torch.autocast('cuda'):
        for index in range(4):
            if index == 2:
                module.weight.add_(0.125)
            if index == 3:
                module.bias = torch.nn.Parameter(module.bias + 0.25)
            actual = module(x)
            expected = F.linear(x, module.weight, module.bias)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            maximum = max(maximum, (actual-expected).abs().max().item())
        state = _weights.get()
        return {'max_abs': maximum, 'cache_hits': state.hits, 'cache_misses': state.misses}


def check_pointwise():
    import torch
    from sam3d_objects.conditioning_cache import sampling_cache
    from sam3d_objects.model.backbone.tdfy_dit.modules import pointwise_rocm
    torch.manual_seed(123)
    results = []
    with torch.no_grad(), sampling_cache('sample_sparse_structure'):
        for tokens in (1, 4, 17, 513, 4096):
            x = torch.randn(1, tokens, 1024, device='cuda')
            h = torch.randn_like(x, dtype=torch.float16)
            for magnitude in (1., 2**-14):
                scale = torch.randn(1, 1024, device='cuda', dtype=torch.float16) * magnitude
                shift = torch.randn_like(scale) * magnitude
                cases = [('modulate', pointwise_rocm.modulate(x, scale, shift),
                          x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)),
                         ('gated_add', pointwise_rocm.gated_add(x, h, scale),
                          x + h * scale.unsqueeze(1))]
                for kind, actual, expected in cases:
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    results.append({'kind': kind, 'tokens': tokens, 'magnitude': magnitude,
                                    'backend': 'triton' if pointwise_rocm._eligible(x, (scale,), False) else 'torch',
                                    'max_abs': (actual-expected).abs().max().item()})
    x.requires_grad_(True)
    with sampling_cache('sample_sparse_structure'):
        pointwise_rocm.modulate(x, scale, shift).sum().backward()
    assert x.grad is not None, 'Gradient-enabled calls must retain PyTorch autograd'
    return results


def check_layout():
    import torch
    import torch.nn.functional as F
    from sam3d_objects.conditioning_cache import sampling_cache, _weights
    from sam3d_objects.linear_rocm import InferenceLinear
    torch.manual_seed(123)
    results = []
    for cin, cout in ((1024, 4096), (4096, 1024)):
        module = InferenceLinear(cin, cout).cuda().half().eval().requires_grad_(False)
        x = torch.randn(513, cin, device='cuda', dtype=torch.float16) * 0.1
        maximum = reference_maximum = 0.
        with torch.no_grad(), sampling_cache('sample_slat'), torch.autocast('cuda'):
            for index in range(3):
                if index == 2:
                    module.weight.add_(0.01)
                actual = module(x)
                row_major = F.linear(x, module.weight, module.bias)
                torch.testing.assert_close(actual, row_major, atol=2e-3, rtol=2e-3)
                reference = (x[:32].cpu().double() @ module.weight.cpu().double().t()
                             + module.bias.cpu().double())
                values = actual[:32].cpu().double()
                torch.testing.assert_close(values, reference, atol=2e-3, rtol=2e-3)
                maximum = max(maximum, (actual-row_major).abs().max().item())
                reference_maximum = max(reference_maximum, (values-reference).abs().max().item())
            state = _weights.get()
            results.append({'input_channels': cin, 'output_channels': cout,
                            'max_abs_vs_row_major': maximum, 'max_abs_vs_float64': reference_maximum,
                            'cache_hits': state.hits if state is not None else 0,
                            'cache_misses': state.misses if state is not None else 0,
                            'atol': 2e-3, 'rtol': 2e-3})
    return results


def check_fused_gelu():
    import os
    from unittest.mock import patch
    import torch
    import torch.nn.functional as F
    from sam3d_objects.conditioning_cache import sampling_cache
    from sam3d_objects.linear_rocm import linear_gelu
    torch.manual_seed(123)
    module = torch.nn.Linear(1024, 4096).cuda().half().eval().requires_grad_(False)
    results = []
    with patch.dict(os.environ, SAM3D_FUSED_GELU='1', SAM3D_CACHE_LAYOUT='1'):
        for tokens in (513, 2049):
            x = torch.randn(tokens, 1024, device='cuda', dtype=torch.float16)
            indices = torch.linspace(0, tokens-1, 32).long()
            with torch.no_grad(), sampling_cache('sample_slat'), torch.autocast('cuda'):
                actual = linear_gelu(module, x)
                assert actual is not None, 'Expected fused latent MLP'
                reference = (x.cpu()[indices].double() @ module.weight.cpu().double().t()
                             + module.bias.cpu().double())
                reference = F.gelu(reference, approximate='tanh')
                values = actual.cpu()[indices].double()
                torch.testing.assert_close(values, reference, atol=2e-3, rtol=2e-3)
                results.append({'tokens': tokens, 'sampled_rows': len(indices),
                                'max_abs_vs_float64': (values-reference).abs().max().item(),
                                'atol': 2e-3, 'rtol': 2e-3})
                with module.register_forward_hook(lambda *args: None):
                    assert linear_gelu(module, x) is None, 'Hooks require the original modules'
        with sampling_cache('sample_slat'), torch.autocast('cuda'):
            assert linear_gelu(module, x) is None, 'Gradients require the original modules'
    return results


def check_compact_context():
    import os
    from unittest.mock import patch
    import torch
    from sam3d_objects.conditioning_cache import compact_zero_context, sampling_cache
    from sam3d_objects.model.backbone.tdfy_dit.modules.attention.modules import MultiHeadAttention
    torch.manual_seed(123)
    results = []
    with patch.dict(os.environ, {'SAM3D_CACHE_CROSS_ATTN': '1'}):
        for dtype in (torch.float16, torch.float32):
            for keys in (31, 33, 7528, 8192):
                model = MultiHeadAttention(1024, 16, type='cross', qk_rms_norm=True).cuda().eval()
                x = torch.randn(1, 17, 1024, device='cuda')
                context = torch.zeros(1, keys, 1024, device='cuda', dtype=dtype)
                with torch.no_grad(), torch.autocast('cuda'), sampling_cache('sample_sparse_structure'):
                    for mutation in (False, True):
                        if mutation:
                            model.to_kv.bias.add_(0.01)
                        with patch.dict(os.environ, {'SAM3D_COMPACT_CFG': '0'}):
                            expected = model(x, context)
                        with patch.dict(os.environ, {'SAM3D_COMPACT_CFG': '1'}):
                            assert compact_zero_context(model.to_kv, context).shape == (1, 1, 1024)
                            actual = model(x, context)
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    with patch.dict(os.environ, {'SAM3D_COMPACT_CFG': '1'}):
                        context[0, -1, -1] = 1
                        assert compact_zero_context(model.to_kv, context) is context
                        context.zero_()
                        model.to_kv.weight[0, 0] = float('inf')
                        assert compact_zero_context(model.to_kv, context) is context
                results.append({'dtype': str(dtype), 'keys': keys, 'max_abs': 0.,
                                'mutation_and_nonfinite_fallback': True})
        context = torch.zeros(1, 31, 1024, device='cuda')
        model = MultiHeadAttention(1024, 16, type='cross', qk_rms_norm=True).cuda().eval()
        x = torch.randn(1, 4, 1024, device='cuda')
        x[0, 0, 0] = float('inf')
        with torch.no_grad(), torch.autocast('cuda'), sampling_cache('sample_sparse_structure'):
            with patch.dict(os.environ, {'SAM3D_COMPACT_CFG': '0'}):
                expected = model(x, context)
            with patch.dict(os.environ, {'SAM3D_COMPACT_CFG': '1'}):
                actual = model(x, context)
                assert compact_zero_context(model.to_kv, context, training=True) is context
            torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)
        with torch.autocast('cuda'), sampling_cache('sample_sparse_structure'):
            assert compact_zero_context(model.to_kv, context) is context
    return results


def check_image_conversion():
    import numpy as np
    from PIL import Image
    from sam3d_objects.pipeline.inference_pipeline import InferencePipeline
    image = np.arange(256, dtype=np.uint8).reshape(8, 8, 4)
    inputs = [image, image[:, ::2], Image.fromarray(image),
              image.astype(np.float32), image.astype(np.float64)]
    results = []
    for source in inputs:
        expected = (np.array(source) / 255).astype(np.float32)
        actual = InferencePipeline.image_to_float(None, source)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        assert not np.shares_memory(np.asarray(source), actual)
        results.append({'type': type(source).__name__, 'shape': list(actual.shape),
                        'bit_exact': True})
    return results


def check_ply_export():
    import tempfile
    import numpy as np
    import torch
    from plyfile import PlyData
    from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian.gaussian_model import Gaussian
    results = []
    torch.manual_seed(123)
    with tempfile.TemporaryDirectory() as directory:
        for dtype in (torch.float32, torch.float64):
            model = Gaussian([-1, -1, -1, 2, 2, 2])
            for name, shape in (('_xyz', (513, 3)), ('_features_dc', (513, 1, 3)),
                                ('_scaling', (513, 3)), ('_rotation', (513, 4)),
                                ('_opacity', (513, 1))):
                setattr(model, name, torch.randn(shape, device='cuda', dtype=dtype))
            model._opacity[:2, 0] = torch.tensor([-float('inf'), float('inf')], device='cuda')
            path = Path(directory) / 'gaussians.ply'
            model.save_ply(str(path))
            data = PlyData.read(str(path))['vertex'].data
            # Check named fields independently, including legal +/-inf opacity
            # logits and float64 inputs that must be converted to float32.
            opacity = model.get_opacity
            expected = {'x': model.get_xyz[:, 0], 'y': model.get_xyz[:, 1],
                        'z': model.get_xyz[:, 2], 'opacity': (opacity / (1-opacity)).log()[:, 0]}
            expected.update({f'f_dc_{i}': model._features_dc[:, 0, i] for i in range(3)})
            expected.update({f'scale_{i}': model.get_scaling[:, i].log() for i in range(3)})
            expected.update({f'rot_{i}': (model._rotation + model.rots_bias)[:, i] for i in range(4)})
            for name in ('nx', 'ny', 'nz'):
                np.testing.assert_array_equal(data[name], np.zeros(513, dtype=np.float32))
            for name, values in expected.items():
                np.testing.assert_array_equal(data[name], values.cpu().numpy().astype(np.float32))
            assert len(data.dtype.names) == 17 and data.dtype.itemsize == 68
            results.append({'dtype': str(dtype), 'rows': len(data), 'max_abs': 0.})
    return results


def check_golden(root):
    import torch
    tests = root / 'tests'
    sys.path.insert(0, str(tests))
    from helpers import load_impl, run_pipeline_case, canonical_order
    impl = load_impl()
    assert impl.name == 'spconv_triton', 'Must test the ROCm replacement backend'
    manifest = dict(line.split(maxsplit=1)[::-1]
                    for line in (tests / 'data/MANIFEST.sha256').read_text().splitlines())
    results = []
    for filename in ['golden_conv.pt', 'golden_fp16.pt', 'golden_wide_conv.pt', 'golden_wide_fp16.pt']:
        path = tests / 'data' / filename
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == manifest[f'data/{filename}'], f'Fixture checksum mismatch: {path}'
        # Only load fixtures from a trusted checkout: these contain pickled metadata.
        for case in torch.load(path, map_location='cpu', weights_only=False)['cases']:
            if case.get('kind') != 'pipeline' or '3d' not in case['id']:
                continue
            # The separate tight fp16 fixtures use CUDA FP32 references, with bounds
            # derived from CUDA fp16 rounding. Skip the older loose fp16 fixtures.
            if filename == 'golden_conv.pt' and case['dtype'] != 'float32':
                continue
            case = {**case, 'training': False}
            with torch.no_grad():
                out = run_pipeline_case(impl, case)['out']
            order = canonical_order(out.indices, out.spatial_shape)
            expected = case['expect']
            assert list(out.spatial_shape) == list(expected['out_spatial_shape'])
            assert torch.equal(out.indices[order].cpu(), expected['out_indices'])
            actual = out.features[order].float().cpu()
            ref = expected['out_features'].float()
            torch.testing.assert_close(actual, ref, rtol=0, atol=expected['atol_out'])
            error = actual - ref
            result = {'id': case['id'], 'dtype': case['dtype'], 'mode': 'eval',
                      'fixture_sha256': digest,
                      'max_abs': error.abs().max().item(),
                      'rms': error.square().mean().sqrt().item(),
                      'reference_rms': ref.square().mean().sqrt().item(),
                      'atol': float(expected['atol_out'])}
            print(json.dumps(result), flush=True)
            results.append(result)
    assert results, 'No reference cases found'
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--golden-root', type=Path, help='trusted pinned spconv-triton checkout')
    parser.add_argument('--output', type=Path, default=Path('/ryzers/output/numerics.json'))
    args = parser.parse_args()
    import torch
    assert torch.version.hip and torch.cuda.is_available(), 'Requires ROCm GPU'
    report = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'rocm': torch.version.hip, 'attention': check_attention(),
              'triton_attention': check_triton_attention(),
              'normalization': check_normalization(), 'linear_cache': check_linear_cache(),
              'pointwise': check_pointwise(), 'layout': check_layout(),
              'fused_gelu': check_fused_gelu(),
              'ply_export': check_ply_export(), 'image_conversion': check_image_conversion(),
              'compact_context': check_compact_context()}
    if args.golden_root:
        report['cuda_sparse_references'] = check_golden(args.golden_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'PASS: numerical checks; report: {args.output}')


if __name__ == '__main__':
    main()
