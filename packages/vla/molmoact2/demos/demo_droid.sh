#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# DROID open-loop replay: run MolmoAct2-DROID on a dataset episode and write a scene
# video plus a ground-truth vs prediction action plot to /outputs.
#
#   ryzers run /ryzers/demo_droid.sh              # random episode
#   EPISODE=42 ryzers run /ryzers/demo_droid.sh   # fixed episode
set -euo pipefail
exec python /ryzers/droid_openloop_demo.py
