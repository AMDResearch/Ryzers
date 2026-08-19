# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""torch.load compatibility shim for LIBERO's pickled assets.

PyTorch >= 2.6 defaults torch.load to weights_only=True, rejecting LIBERO's pickled
init-states. Restore weights_only=False (trusted assets shipped in the image). Applied via
a startup .pth (covers any policy's eval script) and from sim_libero.__init__.
"""


def patch_torch_load():
    try:
        import torch
    except ImportError:
        return
    if getattr(torch.load, "_sim_libero_patched", False):
        return
    _orig = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig(*args, **kwargs)

    _load._sim_libero_patched = True
    torch.load = _load
