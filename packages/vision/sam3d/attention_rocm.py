# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""FP16 streaming attention for large blocks in sparse-latent inference."""
from functools import lru_cache
import os

import torch
import triton
import triton.language as tl

from sam3d_objects.conditioning_cache import sampling_stage


@triton.jit
def _attention_forward(Q, K, V, O, nq, nk,
                       stride_q: tl.constexpr, stride_k: tl.constexpr,
                       stride_v: tl.constexpr,
                       ROWS: tl.constexpr = 128, KEYS: tl.constexpr = 32):
    head = tl.program_id(1)
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    columns = tl.arange(0, KEYS)
    channels = tl.arange(0, 64)
    q = tl.load(Q + rows[:, None] * stride_q + head * 64 + channels[None, :],
                rows[:, None] < nq, other=0)
    maximum = tl.full((ROWS,), float('-inf'), tl.float32)
    denominator = tl.zeros((ROWS,), tl.float32)
    accumulator = tl.zeros((ROWS, 64), tl.float32)
    # Online softmax avoids materializing the query-by-key matrix. Token counts
    # stay runtime arguments so different masks reuse the compiled kernel.
    for start in range(0, nk, KEYS):
        keys = start + columns
        k = tl.load(K + keys[None, :] * stride_k + head * 64 + channels[:, None],
                    keys[None, :] < nk, other=0)
        v = tl.load(V + keys[:, None] * stride_v + head * 64 + channels[None, :],
                    keys[:, None] < nk, other=0)
        scores = tl.dot(q, k) * 0.18033688011112042  # log2(e) / sqrt(64)
        scores = tl.where(keys[None, :] < nk, scores, float('-inf'))
        next_maximum = tl.maximum(maximum, tl.max(scores, 1))
        probabilities = tl.exp2(scores - next_maximum[:, None])
        correction = tl.exp2(maximum - next_maximum)
        denominator = denominator * correction + tl.sum(probabilities, 1)
        accumulator = accumulator * correction[:, None]
        accumulator = tl.dot(probabilities.to(v.dtype), v, accumulator)
        maximum = next_maximum
    tl.store(O + (rows[:, None] * 16 + head) * 64 + channels[None, :],
             accumulator / denominator[:, None], rows[:, None] < nq)


@lru_cache(maxsize=None)
def _supported_device(device):
    arch = getattr(torch.cuda.get_device_properties(device), 'gcnArchName', '').split(':')[0]
    return arch in ('gfx1151', 'gfx1201')


def can_use_attention(q, k, v):
    """Accept only the validated inference geometry; keep SDPA for other cases."""
    return (sampling_stage() == 'sample_slat'
            and os.environ.get('SAM3D_TRITON_ATTN', '1') != '0'
            and not torch.is_grad_enabled()
            and all(x.ndim == 3 and x.shape[1:] == (16, 64)
                    and x.dtype == torch.float16 and x.device == q.device
                    and x.stride(1) == 64 and x.stride(2) == 1 for x in (q, k, v))
            and q.shape[0] >= 2048 and k.shape[0] >= 2048
            and k.shape[0] == v.shape[0]
            and q.device.type == 'cuda' and _supported_device(q.device))


@lru_cache(maxsize=None)
def _configuration(device):
    arch = getattr(torch.cuda.get_device_properties(device), 'gcnArchName', '').split(':')[0]
    return (64, 2) if arch == 'gfx1151' else (32, 0)


def attention(q, k, v):
    """Return [queries, 16, 64]; caller checks eligibility and disables gradients."""
    out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    keys, waves = _configuration(q.device)
    with torch.cuda.device(q.device):
        _attention_forward[(triton.cdiv(q.shape[0], 128), 16)](
            q, k, v, out, q.shape[0], k.shape[0], q.stride(0), k.stride(0), v.stride(0),
            KEYS=keys, num_warps=4, num_stages=1, matrix_instr_nonkdim=16, waves_per_eu=waves)
    return out
