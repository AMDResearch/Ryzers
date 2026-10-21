# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Compute independent attention blocks without a quadratic block-diagonal mask."""
import torch
import torch.nn.functional as F


def masked_sdpa(q, k, v, q_seqlen, kv_seqlen):
    """Preserve upstream's concatenated [1, tokens, heads, channels] interface."""
    if q.shape[0] != 1 or k.shape[0] != 1 or v.shape[0] != 1:
        raise ValueError('Expected one concatenated batch of attention blocks')
    if (len(q_seqlen) != len(kv_seqlen) or sum(q_seqlen) != q.shape[1]
            or sum(kv_seqlen) != k.shape[1] or k.shape[1] != v.shape[1]):
        raise ValueError('Attention block lengths do not match input tensors')
    outputs = []
    q_start = kv_start = 0
    for q_len, kv_len in zip(q_seqlen, kv_seqlen):
        # Each block is independent. Omitting its all-zero additive mask also
        # lets PyTorch select fused attention on gfx1151 and gfx1201.
        out = F.scaled_dot_product_attention(
            q[:, q_start:q_start+q_len].permute(0, 2, 1, 3),
            k[:, kv_start:kv_start+kv_len].permute(0, 2, 1, 3),
            v[:, kv_start:kv_start+kv_len].permute(0, 2, 1, 3),
            dropout_p=0.0, is_causal=False,
        )
        outputs.append(out.permute(0, 2, 1, 3)[0])
        q_start += q_len
        kv_start += kv_len
    return torch.cat(outputs, dim=0) if outputs else q.new_empty((0, q.shape[2], v.shape[3]))
