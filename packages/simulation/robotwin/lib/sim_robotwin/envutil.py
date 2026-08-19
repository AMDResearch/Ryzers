# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Env-var readers that treat an empty string as unset (ryzers passes unset knobs as `-e VAR=`)."""
import os


def env_str(key, default):
    val = os.environ.get(key)
    return val if val not in (None, "") else default


def env_int(key, default):
    return int(env_str(key, str(default)))


def env_float(key, default):
    return float(env_str(key, str(default)))
