# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""torch.load compatibility shim for LIBERO's pickled assets.

LIBERO stores per-task init-states as numpy-backed pickled tensors and loads them via
torch.load. PyTorch >= 2.6 flipped the torch.load default to weights_only=True, which
rejects those pickles. Restore the pre-2.6 default so the LIBERO assets baked into the
image load. Safe here: the files are trusted assets shipped in the simulator image.

Applied globally via sitecustomize.py (so the upstream fastwam eval script, which does not
import sim_libero, is covered) and also from sim_libero.__init__ for direct library use.
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
