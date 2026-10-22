# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Inference normalization kernels for the validated Radeon sampling paths."""
from functools import lru_cache
import os

import torch
import triton
import triton.language as tl

from sam3d_objects.conditioning_cache import sampling_stage


@triton.jit
def _rms_kernel(X, G, Y, sb: tl.constexpr, sn: tl.constexpr,
                sh: tl.constexpr, n, h: tl.constexpr, scale: tl.constexpr):
    rows = tl.program_id(0) * 64 + tl.arange(0, 64)
    batch = tl.program_id(1)
    channels = tl.arange(0, 64)
    tokens, heads = rows // h, rows % h
    x = tl.load(X + batch * sb + tokens[:, None] * sn + heads[:, None] * sh
                + channels[None, :], rows[:, None] < n * h, other=0).to(tl.float32)
    norm = tl.maximum(tl.sqrt(tl.sum(x * x, 1)), 1e-12)
    y = tl.div_rn(x, norm[:, None])
    gamma = tl.load(G + heads[:, None] * 64 + channels[None, :]).to(tl.float32)
    y = y * gamma
    y = y * scale
    tl.store(Y + (batch * n * h + rows[:, None]) * 64 + channels[None, :],
             y, rows[:, None] < n * h)


@triton.jit
def _dense_rms_tail(X, Norm, G, Y, sb: tl.constexpr, sn: tl.constexpr,
                    sh: tl.constexpr, n, h: tl.constexpr, scale: tl.constexpr):
    rows = tl.program_id(0) * 64 + tl.arange(0, 64)
    batch = tl.program_id(1)
    channels = tl.arange(0, 64)
    tokens, heads = rows // h, rows % h
    x = tl.load(X + batch * sb + tokens[:, None] * sn + heads[:, None] * sh
                + channels[None, :], rows[:, None] < n * h, other=0).to(tl.float32)
    norm = tl.load(Norm + batch * n * h + rows, rows < n * h, other=1)
    y = tl.div_rn(x, tl.maximum(norm, 1e-12)[:, None])
    gamma = tl.load(G + heads[:, None] * 64 + channels[None, :]).to(tl.float32)
    y = y * gamma
    y = y * scale
    tl.store(Y + (batch * n * h + rows[:, None]) * 64 + channels[None, :],
             y, rows[:, None] < n * h)


@triton.jit
def _layer_kernel(X, W, B, Y, c: tl.constexpr, eps: tl.constexpr,
                  affine: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    channels = tl.arange(0, block)
    x = tl.load(X + row * c + channels, channels < c, other=0).to(tl.float32)
    mean = tl.sum(x, 0) / c
    centered = tl.where(channels < c, x - mean, 0.)
    variance = tl.sum(centered * centered, 0) / c
    y = centered * tl.rsqrt(variance + eps)
    if affine:
        weight = tl.load(W + channels, channels < c, other=0).to(tl.float32)
        bias = tl.load(B + channels, channels < c, other=0).to(tl.float32)
        y = y * weight + bias
    tl.store(Y + row * c + channels, y, channels < c)


def _rms(x, gamma, scale, kernel):
    out = torch.empty(x.shape, device=x.device, dtype=torch.float16)
    batch = x.shape[0] if x.ndim == 4 else 1
    n, h, _ = x.shape[-3:]
    sb = x.stride(0) if x.ndim == 4 else 0
    with torch.cuda.device(x.device):
        kernel[(triton.cdiv(n * h, 64), batch)](
            x, gamma, out, sb, x.stride(-3), x.stride(-2), n, h, scale,
            num_warps=4, enable_fp_fusion=False)
    return out


def sparse_rms(x, gamma, scale):
    return _rms(x, gamma, scale, _rms_kernel)


def dense_rms(x, gamma, scale):
    # Autocast runs upstream's F.normalize reduction in FP32. Keep that exact
    # PyTorch reduction and fuse only the clamp, division, affine and cast.
    norm = torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=True)
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    n, h, _ = x.shape[-3:]
    with torch.cuda.device(x.device):
        _dense_rms_tail[(triton.cdiv(n * h, 64), x.shape[0])](
            x, norm, gamma, out, x.stride(0), x.stride(1), x.stride(2), n, h, scale,
            num_warps=4, enable_fp_fusion=False)
    return out


def layer_norm(x, weight, bias, eps):
    c = x.shape[-1]
    out = torch.empty_like(x)
    with torch.cuda.device(x.device):
        _layer_kernel[(x.numel() // c,)](
            x, weight if weight is not None else x, bias if bias is not None else x,
            out, c, eps, weight is not None, triton.next_power_of_2(c),
            num_warps=4, enable_fp_fusion=False)
    return out


@lru_cache(maxsize=None)
def _supported_device(device):
    arch = getattr(torch.cuda.get_device_properties(device), 'gcnArchName', '').split(':')[0]
    return arch in ('gfx1151', 'gfx1201')


@lru_cache(maxsize=None)
def _dense_supported_device(device):
    arch = getattr(torch.cuda.get_device_properties(device), 'gcnArchName', '').split(':')[0]
    return arch == 'gfx1201'


def _enabled(module, x):
    return (sampling_stage() == 'sample_slat'
            and os.environ.get('SAM3D_TRITON_NORM', '1') != '0'
            and not module.training and not torch.is_grad_enabled()
            and x.dtype == torch.float16 and x.device.type == 'cuda'
            and _supported_device(x.device))


def can_use_rms(module, x):
    return (_enabled(module, x)
            and x.ndim in (3, 4) and x.shape[-3] > 0
            and x.shape[-2:] == (16, 64)
            and module.gamma.shape == (16, 64)
            and module.gamma.dtype == torch.float32
            and module.gamma.device == x.device and module.gamma.is_contiguous())


def can_use_dense_rms(module, x):
    return (sampling_stage() == 'sample_sparse_structure'
            and os.environ.get('SAM3D_TRITON_NORM', '1') != '0'
            and os.environ.get('SAM3D_TRITON_DENSE_NORM', '1') != '0'
            and not module.training and not torch.is_grad_enabled()
            and x.device.type == 'cuda' and x.dtype == torch.float16
            and torch.is_autocast_enabled('cuda')
            and torch.get_autocast_dtype('cuda') == torch.float16
            and x.ndim == 4 and x.shape[0] == 1 and x.shape[1] > 0
            and x.shape[-2:] == (16, 64) and x.stride(-1) == 1
            and module.gamma.shape == (16, 64) and module.gamma.dtype == torch.float32
            and module.gamma.device == x.device and module.gamma.is_contiguous()
            and _dense_supported_device(x.device))


def can_use_separate_qkv(module, x):
    return os.environ.get('SAM3D_SEPARATE_QKV', '1') != '0' and can_use_rms(module, x)


def can_use_layer(module, x):
    return (_enabled(module, x)
            and x.ndim == 2 and x.shape[0] >= 512
            and x.shape[1] in (1024, 2048) and x.is_contiguous()
            and module.normalized_shape == (x.shape[1],)
            and ((module.weight is None and module.bias is None)
                 or (module.weight is not None and module.bias is not None
                     and all(p.dtype == torch.float32 and p.device == x.device
                             and p.is_contiguous() for p in (module.weight, module.bias)))))
