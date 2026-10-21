# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Bounded reuse of immutable image conditioning within one sampling stage."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import os

import torch


_current = ContextVar('sam3d_conditioning_cache', default=None)


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
        if output_version is not None and len(self.entries) < 256 and self.bytes + size <= 3 * 1024**3:
            # Hold the input as well: Python must not recycle its identity.
            self.entries[key] = (tensor, output, output_version)
            self.bytes += size
        return output


@contextmanager
def sampling_cache():
    state = _Cache() if os.environ.get('SAM3D_CACHE_CROSS_ATTN', '1') != '0' else None
    token = _current.set(state)
    try:
        yield state
    finally:
        _current.reset(token)
        if state is not None:
            state.entries.clear()


def cache_sampling(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with sampling_cache():
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
