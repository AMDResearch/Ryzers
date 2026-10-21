#!/usr/bin/env python3
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Check attention against float64 math and sparse convolution against CUDA fixtures."""
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
              'rocm': torch.version.hip, 'attention': check_attention()}
    if args.golden_root:
        report['cuda_sparse_references'] = check_golden(args.golden_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'PASS: numerical checks; report: {args.output}')


if __name__ == '__main__':
    main()
