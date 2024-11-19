# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from setuptools import setup, find_packages
import os

# Path to the directory of this setup.py file
setup_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the ryzers package
ryzer_dir = os.path.join(setup_dir, 'ryzers')

# Path to the __init__.py file in the ryzers package
init_file = os.path.join(ryzer_dir, '__init__.py')

if os.name == 'nt':  # Windows
    setup_dir = setup_dir.replace('\\', '\\\\')


# Read the current contents of the file
with open(init_file, 'r') as file:
    lines = file.readlines()

# Modify the line starting with RYZER_PACKAGES_PATH =
with open(init_file, 'w') as file:
    for line in lines:
        if line.startswith('RYZER_PACKAGES_PATH ='):
            line = f'RYZER_PACKAGES_PATH = "{setup_dir}"\n'
        file.write(line)

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

