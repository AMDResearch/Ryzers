# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Preserve intermediate rounding while fusing structure-sampler elementwise ops."""
import os

import torch
import triton
import triton.language as tl

from sam3d_objects.conditioning_cache import sampling_stage
from .norm_rocm import _supported_device


@triton.jit
def _modulate(X, S, T, Y, n, channels: tl.constexpr):
    index = tl.program_id(0) * 4096 + tl.arange(0, 4096)
    channel = index % channels
    x = tl.load(X + index, index < n, other=0)
    scale = tl.load(S + channel).to(tl.float32)
    shift = tl.load(T + channel).to(tl.float32)
    # PyTorch rounds the scalar addition to FP16 before promoting the product.
    coefficient = (scale + 1.).to(tl.float16).to(tl.float32)
    result = x * coefficient
    result = result + shift
    tl.store(Y + index, result, index < n)


@triton.jit
def _gated_add(X, H, G, Y, n, channels: tl.constexpr):
    index = tl.program_id(0) * 4096 + tl.arange(0, 4096)
    channel = index % channels
    x = tl.load(X + index, index < n, other=0)
    h = tl.load(H + index, index < n, other=0).to(tl.float32)
    gate = tl.load(G + channel).to(tl.float32)
    product = (h * gate).to(tl.float16).to(tl.float32)
    tl.store(Y + index, x + product, index < n)


def _eligible(x, tensors, training):
    return (not training and not torch.is_grad_enabled()
            and sampling_stage() == 'sample_sparse_structure'
            and os.environ.get('SAM3D_TRITON_POINTWISE', '1') != '0'
            and type(x) is torch.Tensor and x.device.type == 'cuda'
            and x.ndim == 3 and x.shape[0] == 1 and x.shape[1] >= 1
            and x.shape[-1] == 1024 and x.dtype == torch.float32
            and x.is_contiguous() and _supported_device(x.device)
            and all(type(t) is torch.Tensor and t.device == x.device
                    and t.dtype == torch.float16 and t.is_contiguous() for t in tensors))


def modulate(x, scale, shift, training=False):
    if (_eligible(x, (scale, shift), training)
            and scale.shape == shift.shape == (1, 1024)):
        out = torch.empty_like(x)
        with torch.cuda.device(x.device):
            _modulate[(triton.cdiv(x.numel(), 4096),)](
                x, scale, shift, out, x.numel(), 1024,
                num_warps=4, enable_fp_fusion=False)
        return out
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def gated_add(x, h, multiplier, training=False):
    if (_eligible(x, (h, multiplier), training)
            and h.shape == x.shape and multiplier.shape == (1, 1024)):
        out = torch.empty_like(x)
        with torch.cuda.device(x.device):
            _gated_add[(triton.cdiv(x.numel(), 4096),)](
                x, h, multiplier, out, x.numel(), 1024,
                num_warps=4, enable_fp_fusion=False)
        return out
    return x + h * multiplier.unsqueeze(1)
