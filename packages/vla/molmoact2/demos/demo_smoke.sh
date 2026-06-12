#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Full-model smoke: load MolmoAct2-DROID and run one action prediction on ROCm.
# First run downloads ~22 GB into the HF cache.
set -euo pipefail
exec python /ryzers/model_smoke.py
