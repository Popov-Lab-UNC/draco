#!/bin/bash

#SBATCH --job-name=draco_dynamics

#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1

##SBATCH --partition=general
#SBATCH -n 1
#SBATCH --cpus-per-task=16

#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=test_2_%j.log
#SBATCH --error=test_2_%j.err
#SBATCH --mail-user=shuhang@unc.edu
#SBATCH --mail-type=ALL

set -euo pipefail

module load cuda
module load apptainer
module load autodock-gpu
module load autogrid

export PIXI_HOME=/work/users/s/h/shuhang/.pixi
export PATH="$PIXI_HOME/bin:$PATH"

cd /work/users/s/h/shuhang/draco

# Normalize CPU allocation so both bash (-u) and draco see a safe value.
CPUS="${SLURM_CPUS_PER_TASK:-16}"
export SLURM_CPUS_PER_TASK="$CPUS"

# Avoid hidden thread oversubscription inside numeric libraries.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

pixi run draco \
  --mode sar \
  --steps pocket docking scoring \
  --protein-pdb       "6WAU-SAR-30mar2026/frames/frame_0007.pdb" \
  --ligand-csv        "/users/s/h/shuhang/PHF19/phf19_data_30mar2026.csv" \
  --docking-engine    adgpu_gnina \
  --gnina-binary      "apptainer run --nv /work/users/s/h/shuhang/gnina_build/gnina.sif" \
  --cnn-scoring       refinement \
  --scoring-method    cnn_vs \
  --n-cpus            "$CPUS" \
  --output-dir        "6WAU-SAR-4may2026"