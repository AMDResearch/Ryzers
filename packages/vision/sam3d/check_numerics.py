#!/usr/bin/env python3
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Check ROCm attention, normalization, parameter reuse and CUDA sparse fixtures."""
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
              'normalization': check_normalization(), 'linear_cache': check_linear_cache()}
    if args.golden_root:
        report['cuda_sparse_references'] = check_golden(args.golden_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'PASS: numerical checks; report: {args.output}')


if __name__ == '__main__':
    main()
