#/!bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

source env.sh
source $SPSDEMO_VENV/bin/activate

cd $SPSDEMO_PATH && \
    xdg-open ./patches/index.html && \
    ryzers run --name smolvlm /ryzers/demo_smolvlm.sh
