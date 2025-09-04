# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT


RYZERS_DEFAULT_INIT_IMAGE = "ubuntu:24.04"
RYZERS_DEFAULT_RUN_FLAGS = "-it --rm --shm-size 16G --cap-add=SYS_PTRACE  --network=host --ipc=host"

# This gets populated with path to setup.py during pip install
RYZERS_PACKAGES_PATH = None
