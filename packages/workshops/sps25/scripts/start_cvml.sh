#/!bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

source env.sh
source $SPSDEMO_VENV/bin/activate

cd $SPSDEMO_PATH && \
    xdg-open http://127.0.0.1:8080 && \
    ryzers run --name cvml /ryzers/demo_ros.sh
