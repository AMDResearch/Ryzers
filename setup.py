# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from setuptools import setup, find_packages

setup(
    name="ryzers",
    version="1.0",
    description="A tool to Manage and Build Dockerfiles for Ryzen AI Applications.",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'ryzers = ryzers.entrypoint:main',           
        ],
    },
    install_requires=[
        'pyyaml',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
