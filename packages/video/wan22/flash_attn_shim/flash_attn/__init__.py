"""Small SDPA-backed `flash_attn` shim for ROCm bring-up."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

__version__ = "0.0.0+sdpa-shim"


def _to_bnsd(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError(f"expected (batch, seq, heads, dim), got {tuple(x.shape)}")
    return x.transpose(1, 2).contiguous()


def _from_bnsd(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(1, 2).contiguous()


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
):
    if window_size != (-1, -1):
        raise NotImplementedError("sliding-window attention is not implemented in the ROCm shim")
    if alibi_slopes is not None:
        raise NotImplementedError("ALiBi is not implemented in the ROCm shim")
    if return_attn_probs:
        raise NotImplementedError("attention-probability return is not implemented in the ROCm shim")
    out = F.scaled_dot_product_attention(
        _to_bnsd(q),
        _to_bnsd(k),
        _to_bnsd(v),
        attn_mask=None,
        dropout_p=dropout_p if torch.is_grad_enabled() else 0.0,
        is_causal=causal,
        scale=softmax_scale,
    )
    return _from_bnsd(out)


def flash_attn_qkvpacked_func(qkv: torch.Tensor, **kwargs):
    if qkv.dim() != 5 or qkv.size(2) != 3:
        raise ValueError(f"expected (batch, seq, 3, heads, dim), got {tuple(qkv.shape)}")
    q, k, v = qkv.unbind(dim=2)
    return flash_attn_func(q, k, v, **kwargs)


def flash_attn_kvpacked_func(q: torch.Tensor, kv: torch.Tensor, **kwargs):
    if kv.dim() != 5 or kv.size(2) != 2:
        raise ValueError(f"expected (batch, seq, 2, heads, dim), got {tuple(kv.shape)}")
    k, v = kv.unbind(dim=2)
    return flash_attn_func(q, k, v, **kwargs)


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    **kwargs,
):
    cu_q = cu_seqlens_q.detach().to(dtype=torch.int64, device="cpu").tolist()
    cu_k = cu_seqlens_k.detach().to(dtype=torch.int64, device="cpu").tolist()
    out = torch.empty((q.shape[0], q.shape[1], v.shape[2]), dtype=q.dtype, device=q.device)
    for batch_idx in range(len(cu_q) - 1):
        sq, eq = cu_q[batch_idx], cu_q[batch_idx + 1]
        sk, ek = cu_k[batch_idx], cu_k[batch_idx + 1]
        qi = q[sq:eq].transpose(0, 1).unsqueeze(0).contiguous()
        ki = k[sk:ek].transpose(0, 1).unsqueeze(0).contiguous()
        vi = v[sk:ek].transpose(0, 1).unsqueeze(0).contiguous()
        oi = F.scaled_dot_product_attention(
            qi,
            ki,
            vi,
            dropout_p=dropout_p if torch.is_grad_enabled() else 0.0,
            is_causal=causal,
            scale=softmax_scale,
        )
        out[sq:eq] = oi.squeeze(0).transpose(0, 1).contiguous()
    return out


flash_attn_unpadded_func = flash_attn_varlen_func

