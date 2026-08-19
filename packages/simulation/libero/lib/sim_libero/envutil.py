# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Env-var readers that treat an empty string as unset.

ryzers run passes optional knobs as `-e VAR=${VAR:-}` (empty when unset), so plain
os.environ.get would return "" and break int()/float(); these fall back to the default.
"""
import os


def env_str(key, default):
    val = os.environ.get(key)
    return val if val not in (None, "") else default


def env_int(key, default):
    return int(env_str(key, str(default)))


def env_float(key, default):
    return float(env_str(key, str(default)))
