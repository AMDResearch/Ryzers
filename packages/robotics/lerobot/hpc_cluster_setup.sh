#!/bin/bash

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# This script sets up the environment for training a policy using the LeRobot framework on the AMD AI & HPC cluster.
# It creates a Python virtual environment, installs necessary packages, and submits an example job to the scheduler.
# AMD AI & HPC cluster: https://www.amd.com/en/corporate/university-program/ai-hpc-cluster.html

cd $WORK
python3.12 -m venv robot_env
source robot_env/bin/activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3

git clone https://github.com/huggingface/lerobot
cd lerobot
git checkout edfebd5
sed -i '67d' pyproject.toml
python -m pip install -e .

python -m pip install transformers pytest

cat << 'EOF' > job.train_pusht_example
#!/bin/bash
#SBATCH -J train_policy           # Job name
#SBATCH -o train_pusht_example.log         # Name of stdout output file (%j expands to jobId)
#SBATCH -N 1                  # Total number of nodes requested
#SBATCH -t 08:00:00           # Run time (hh:mm:ss) - 1.5 hours
#SBATCH -p mi2104x            # Desired partition
module purge
module load hpcfund
source $WORK/robot_env/bin/activate
export HF_HOME="${HOME}/hf_models/" && mkdir -p "$HF_HOME" # mkdir if it doesn't exist
cd $WORK/lerobot
python lerobot/scripts/train.py \
--policy.type=act \
--dataset.repo_id=lerobot/pusht \
--dataset.video_backend=pyav \
--policy.device=cuda
EOF

sbatch job.train_pusht_example
tail -f $WORK/lerobot/train_pusht_example.log
