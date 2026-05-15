#!/bin/bash
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

lsb_release -a 2>/dev/null

python3 -c "
import torch
hip = torch.version.hip or ''
print(f'PyTorch {torch.__version__}  ROCm: {hip}  CUDA: {torch.cuda.is_available()}  Devices: {torch.cuda.device_count()}')

if not hip:
    print('WARNING: ROCm torch missing — attempting restore...')
    import subprocess, sys
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install',
        'torch', 'torchvision', 'torchaudio',
        '--index-url', 'https://download.pytorch.org/whl/rocm7.2/',
        '--force-reinstall', '--no-deps', '-q'
    ])
    # Re-import after restore
    import importlib; importlib.invalidate_caches()
    torch = importlib.import_module('torch')
    hip = torch.version.hip or ''
    print(f'After restore: PyTorch {torch.__version__}  ROCm: {hip}')
    assert hip, f'ROCm torch restore FAILED: {torch.__version__}'
    print('ROCm torch restored successfully')

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('  No GPU (expected at runtime with --device=/dev/kfd)')
"
