#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Capability 1: full-model smoke. Load the real MolmoAct2-DROID checkpoint and
# run one action prediction on ROCm. First run downloads ~22 GB into the mounted
# HF cache.  Usage:  ryzers run /ryzers/demo_smoke.sh
set -euo pipefail
exec python /ryzers/model_smoke.py
