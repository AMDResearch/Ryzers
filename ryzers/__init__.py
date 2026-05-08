# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import os

RYZERS_DEFAULT_INIT_IMAGE = "rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0"
RYZERS_DEFAULT_RUN_FLAGS = "-it --rm --shm-size 16G --cap-add=SYS_PTRACE  --network=host --ipc=host"

# Auto-detect packages path in editable mode
RYZERS_PACKAGES_PATH = None

# If still None, try to detect from __file__
if RYZERS_PACKAGES_PATH is None:
    # In editable install, __file__ points to the source location
    if __file__:
        # Go up from ryzers/__init__.py to the repo root
        _init_path = os.path.dirname(os.path.abspath(__file__))
        _repo_root = os.path.dirname(_init_path)
        if os.path.exists(os.path.join(_repo_root, 'packages')):
            RYZERS_PACKAGES_PATH = _repo_root
