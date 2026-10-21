# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Bounded reuse of frozen linear casts and iGPU MLP layouts during sampling."""
from functools import lru_cache

import torch
import torch.nn.functional as F

from sam3d_objects.conditioning_cache import _weights, sampling_stage


@lru_cache(maxsize=None)
def _architecture(device):
    return getattr(torch.cuda.get_device_properties(device), 'gcnArchName', '').split(':')[0]


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
            weight = cache.get(('column',), self.weight,
                               lambda: self.weight.t().contiguous().t())
            return F.linear(x, weight, self.bias)
        return super().forward(x)
