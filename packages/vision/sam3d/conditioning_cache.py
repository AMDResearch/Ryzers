# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Bounded reuse of image conditioning and frozen parameters per sampling stage."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache, wraps
import os

import torch


_current = ContextVar('sam3d_conditioning_cache', default=None)
_stage = ContextVar('sam3d_sampling_stage', default=None)
_weights = ContextVar('sam3d_weight_cache', default=None)


def sampling_stage():
    return _stage.get()


def _version(tensor):
    # Inference-mode tensors have no mutation counter: use the upstream path.
    try:
        return tensor._version
    except RuntimeError:
        return None


class _Cache:
    def __init__(self):
        self.entries = {}
        self.bytes = 0
        self.hits = 0
        self.misses = 0
        self.max_entries = 256
        self.max_bytes = 3 * 1024**3

    def get(self, key, tensor, compute):
        version = _version(tensor)
        if version is None:
            return compute()
        key = (*key, id(tensor), version)
        entry = self.entries.get(key)
        if entry is not None and _version(entry[1]) == entry[2]:
            self.hits += 1
            return entry[1]
        self.misses += 1
        output = compute()
        size = output.numel() * output.element_size()
        output_version = _version(output)
        if (output_version is not None and len(self.entries) < self.max_entries
                and self.bytes + size <= self.max_bytes):
            # Hold the input as well: Python must not recycle its identity.
            self.entries[key] = (tensor, output, output_version)
            self.bytes += size
        return output


class _WeightCache(_Cache):
    def __init__(self):
        super().__init__()
        self.max_entries = 2048
        self.max_bytes = 2 * 1024**3


@contextmanager
def sampling_cache(stage=None):
    state = _Cache() if os.environ.get('SAM3D_CACHE_CROSS_ATTN', '1') != '0' else None
    reuse_weights = ((stage == 'sample_sparse_structure'
                      and os.environ.get('SAM3D_CACHE_LINEAR', '1') != '0')
                     or (stage == 'sample_slat'
                         and os.environ.get('SAM3D_CACHE_LAYOUT', '1') != '0'))
    weights = _WeightCache() if reuse_weights else None
    weight_token = _weights.set(weights)
    token = _current.set(state)
    stage_token = _stage.set(stage)
    try:
        yield state
    finally:
        _current.reset(token)
        _stage.reset(stage_token)
        _weights.reset(weight_token)
        if weights is not None:
            weights.entries.clear()
        if state is not None:
            state.entries.clear()


def cache_sampling(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with sampling_cache(function.__name__):
            return function(*args, **kwargs)
    return wrapped


def _reuse(key, tensor, compute):
    state = _current.get()
    if state is None or torch.is_grad_enabled() or type(tensor) is not torch.Tensor:
        return compute()
    return state.get(key, tensor, compute)


def conditioning_cast(tensor, dtype):
    return _reuse(('cast', dtype), tensor, lambda: tensor.type(dtype))


def conditioning_zero(tensor):
    return _reuse(('zero',), tensor, lambda: tensor * 0)


@lru_cache(maxsize=None)
def _compact_device(device):
    arch = getattr(torch.cuda.get_device_properties(device), 'gcnArchName', '').split(':')[0]
    return arch in ('gfx1151', 'gfx1201')


def compact_zero_context(module, tensor, training=False):
    """A finite linear projection of zero context has identical keys and values."""
    state = _current.get()
    if (state is None or training or module.training or torch.is_grad_enabled()
            or os.environ.get('SAM3D_COMPACT_CFG', '1') == '0'
            or sampling_stage() not in ('sample_sparse_structure', 'sample_slat')
            or type(tensor) is not torch.Tensor or tensor.device.type != 'cuda'
            or tensor.dtype not in (torch.float16, torch.float32)
            or tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[2] != 1024
            or not 2 <= tensor.shape[1] <= 8192 or _version(tensor) is None
            or not torch.is_autocast_enabled('cuda')
            or torch.get_autocast_dtype('cuda') != torch.float16
            or not _compact_device(tensor.device)):
        return tensor
    from sam3d_objects.linear_rocm import InferenceLinear
    if type(module) not in (torch.nn.Linear, InferenceLinear):
        return tensor
    # Cache CPU booleans so subsequent blocks do not synchronize the GPU.
    zero = state.get(('compact-zero',), tensor, lambda: torch.tensor(
        not bool(torch.count_nonzero(tensor)), device='cpu'))
    if not bool(zero):
        return tensor
    for parameter in (module.weight, module.bias):
        if parameter is None:
            continue
        if (parameter.device != tensor.device or _version(parameter) is None
                or parameter.dtype not in (torch.float16, torch.float32)):
            return tensor
        finite = state.get(('compact-finite',), parameter, lambda p=parameter: torch.tensor(
            bool((torch.isfinite(p) & (p.abs() <= 65504)).all()), device='cpu'))
        if not bool(finite):
            return tensor
    return state.get(('compact-view',), tensor, lambda: tensor[:, :1])


def conditioning_projection(module, tensor, compute=None):
    if compute is None:
        compute = lambda: module(tensor)
    if (module.training or _current.get() is None or torch.is_grad_enabled()
            or type(tensor) is not torch.Tensor):
        return compute()
    parameters = tuple((p, _version(p)) for p in module.parameters())
    if any(version is None for _, version in parameters):
        return compute()
    device_type = tensor.device.type
    key = ('projection', module, parameters, torch.is_autocast_enabled(device_type),
           torch.get_autocast_dtype(device_type))
    return _reuse(key, tensor, compute)
