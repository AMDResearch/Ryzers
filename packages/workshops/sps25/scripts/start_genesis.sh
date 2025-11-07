#/!bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

source env.sh

cd $SPSDEMO_PATH && \
    ryzers run --name genesis "python3 /ryzers/demo_genesis.py"
