#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

source env.sh
source $SPSDEMO_VENV/bin/activate

pushd $SPSDEMO_PATH
ryzers build genesis --name genesis
ryzers build llamacpp smolvlm --name smolvlm
ryzers build xdna ryzenai_cvml ros --name cvml
popd
