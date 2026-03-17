#!/bin/bash
#SBATCH --job-name=test_installation
#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=test_installation_%j.log
#SBATCH --error=test_installation_%j.err
#SBATCH --mail-user=shuhang@unc.edu
#SBATCH --mail-type=ALL

module load cuda

source /work/users/s/h/shuhang/draco/.venv/bin/activate
which python

python -m openmm.testInstallation
