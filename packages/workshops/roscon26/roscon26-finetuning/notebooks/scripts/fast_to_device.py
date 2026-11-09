# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""One HIP copy per dtype instead of one copy per nn.Parameter (MolmoAct2 has ~1300)."""
from collections import defaultdict

_installed = False


def _same_device(a, b):
    import torch

    a, b = torch.device(a), torch.device(b)
    if a.type != b.type:
        return False
    if a.type in ("cuda", "hip"):
        ai = a.index if a.index is not None else torch.cuda.current_device()
        bi = b.index if b.index is not None else torch.cuda.current_device()
        return ai == bi
    return a == b


def bulk_to_device(module, device, dtype=None):
    import torch

    device = torch.device(device)
    unique, aliases = {}, defaultdict(list)
    for t in list(module.parameters()) + list(module.buffers()):
        if _same_device(t.device, device) and (dtype is None or t.dtype == dtype):
            continue
        if not t.is_floating_point() or t.numel() == 0:
            t.data = t.data.to(device=device, dtype=dtype or t.dtype)
            continue
        ptr = t.data_ptr()
        if ptr in unique:
            aliases[ptr].append(t)
        else:
            unique[ptr] = t

    groups = defaultdict(list)
    for t in unique.values():
        groups[dtype or t.dtype].append(t)

    for out_dtype, tensors in groups.items():
        flats = [t.detach().contiguous().to(out_dtype).reshape(-1) for t in tensors]
        packed = torch.cat(flats).to(device)
        off = 0
        for t in tensors:
            n = t.numel()
            # clone so tensors don't share one packed storage (safetensors save needs that)
            owned = packed[off : off + n].view(t.shape).clone()
            old = t.data_ptr()
            t.data = owned
            for alias in aliases.get(old, []):
                alias.data = owned
            off += n
        del packed
        print(f"fast_to_device: {len(tensors)} tensors {out_dtype} -> {device}", flush=True)
    return module


def install():
    """Patch nn.Module.to so Policy(...).to(cuda) uses bulk_to_device."""
    global _installed
    if _installed:
        return
    import torch

    orig = torch.nn.Module.to

    def _to(self, *args, **kwargs):
        try:
            device, cast_dtype, _, fmt = torch._C._nn._parse_to(*args, **kwargs)
        except Exception:
            return orig(self, *args, **kwargs)
        if fmt is not None or device is None:
            return orig(self, *args, **kwargs)
        device = torch.device(device)
        if device.type not in ("cuda", "hip"):
            return orig(self, *args, **kwargs)
        n = sum(1 for _ in self.parameters())
        if n < 32:
            return orig(self, *args, **kwargs)
        return bulk_to_device(self, device, dtype=cast_dtype)

    torch.nn.Module.to = _to
    _installed = True
