# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Env-var readers that treat an empty string as unset.

The ryzers run wrapper passes optional knobs as `-e VAR=${VAR:-}`, i.e. an empty string
when the host did not set them. Plain os.environ.get(key, default) would then return ""
(not the default) and break int()/float() parsing, so these helpers fall back to the
default whenever the value is missing or empty.
"""
import os


def env_str(key, default):
    val = os.environ.get(key)
    return val if val not in (None, "") else default


def env_int(key, default):
    return int(env_str(key, str(default)))


def env_float(key, default):
    return float(env_str(key, str(default)))
