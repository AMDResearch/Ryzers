# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Reuse frozen linear-parameter casts within sparse-structure inference."""
from functools import lru_cache

import torch
import torch.nn.functional as F

from sam3d_objects.conditioning_cache import _weights


@lru_cache(maxsize=None)
def _supported_device(device):
    arch = getattr(torch.cuda.get_device_properties(device), 'gcnArchName', '').split(':')[0]
    return arch in ('gfx1151', 'gfx1201')


class InferenceLinear(torch.nn.Linear):
    def forward(self, x):
        cache = _weights.get()
        if (cache is None or self.training or torch.is_grad_enabled()
                or x.device.type != 'cuda' or self.weight.device != x.device
                or x.dtype not in (torch.float16, torch.float32)
                or self.weight.dtype != torch.float32
                or (self.bias is not None and (self.bias.dtype != torch.float32
                                              or self.bias.device != x.device))
                or not torch.is_autocast_enabled('cuda')
                or torch.get_autocast_dtype('cuda') != torch.float16
                or not _supported_device(x.device)):
            return super().forward(x)
        weight = cache.get(('linear', torch.float16), self.weight,
                           lambda: self.weight.to(dtype=torch.float16))
        bias = None if self.bias is None else cache.get(
            ('linear', torch.float16), self.bias, lambda: self.bias.to(dtype=torch.float16))
        return F.linear(x, weight, bias)
