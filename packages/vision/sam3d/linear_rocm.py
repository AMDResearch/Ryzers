# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Bounded reuse of frozen linear casts and iGPU MLP layouts during sampling."""
from functools import lru_cache
import os

import torch
import torch.nn.functional as F

from sam3d_objects.conditioning_cache import _weights, sampling_stage


@lru_cache(maxsize=None)
def _architecture(device):
    return getattr(torch.cuda.get_device_properties(device), 'gcnArchName', '').split(':')[0]


def _column_weight(weight, cache):
    return cache.get(('column',), weight, lambda: weight.t().contiguous().t())


def linear_gelu(module, x):
    """Use BLASLt's FP16 bias/tanh-GELU epilogue in the latent sampler."""
    if (sampling_stage() != 'sample_slat'
            or os.environ.get('SAM3D_FUSED_GELU', '1') == '0'
            or module.training or torch.is_grad_enabled()
            or module._forward_hooks or module._forward_pre_hooks
            or x.device.type != 'cuda' or x.dtype != torch.float16
            or x.ndim != 2 or x.shape[0] < 512 or x.shape[1] != 1024
            or not x.is_contiguous() or not torch.is_autocast_enabled('cuda')
            or torch.get_autocast_dtype('cuda') != torch.float16
            or module.weight.dtype != torch.float16 or module.weight.shape != (4096, 1024)
            or module.weight.device != x.device or not module.weight.is_contiguous()
            or module.bias is None or module.bias.dtype != torch.float16
            or module.bias.device != x.device or module.bias.shape != (4096,)
            or _architecture(x.device) not in ('gfx1151', 'gfx1201')):
        return None
    weight = module.weight
    if _architecture(x.device) == 'gfx1151':
        cache = _weights.get()
        if cache is None:
            return None
        weight = _column_weight(weight, cache)
    return torch._addmm_activation(module.bias, x, weight.t(), use_gelu=True)


class InferenceLinear(torch.nn.Linear):
    def forward(self, x):
        cache = _weights.get()
        if (cache is None or self.training or torch.is_grad_enabled()
                or x.device.type != 'cuda' or self.weight.device != x.device
                or not torch.is_autocast_enabled('cuda')
                or torch.get_autocast_dtype('cuda') != torch.float16
                or _architecture(x.device) not in ('gfx1151', 'gfx1201')):
            return super().forward(x)
        if (sampling_stage() == 'sample_sparse_structure'
                and x.dtype in (torch.float16, torch.float32)
                and self.weight.dtype == torch.float32
                and (self.bias is None or (self.bias.dtype == torch.float32
                                          and self.bias.device == x.device))):
            weight = cache.get(('linear', torch.float16), self.weight,
                               lambda: self.weight.to(dtype=torch.float16))
            bias = None if self.bias is None else cache.get(
                ('linear', torch.float16), self.bias, lambda: self.bias.to(dtype=torch.float16))
            return F.linear(x, weight, bias)
        if (sampling_stage() == 'sample_slat' and _architecture(x.device) == 'gfx1151'
                and x.ndim == 2 and x.shape[0] >= 512 and x.is_contiguous()
                and x.dtype == self.weight.dtype == torch.float16
                and self.weight.shape in ((4096, 1024), (1024, 4096))
                and (self.bias is None or (self.bias.dtype == torch.float16
                                          and self.bias.device == x.device))):
            weight = _column_weight(self.weight, cache)
            return F.linear(x, weight, self.bias)
        return super().forward(x)
