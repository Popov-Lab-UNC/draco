#!/bin/bash
#SBATCH --job-name=draco_dynamics

#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1

##SBATCH --partition=general
#SBATCH -n 1
#SBATCH --cpus-per-task=16

#SBATCH --mem=32G
#SBATCH --time=01-00:00:00
#SBATCH --output=test_dynamics_%j.log
#SBATCH --error=test_dynamics_%j.err
#SBATCH --mail-user=shuhang@unc.edu
#SBATCH --mail-type=ALL

module load cuda
module load apptainer

export PIXI_HOME=/work/users/s/h/shuhang/.pixi
export PATH="$PIXI_HOME/bin:$PATH"

cd /work/users/s/h/shuhang/draco

# Requires existing frames under OUTPUT_DIR/frames/*.pdb (from a prior MD run).
# Skips explicit-solvent MD; runs Pocketeer + GNINA + optional refinement only.
OUTPUT_DIR="test_dynamics_output_2"

pixi run draco \
  --mode sar \
  --steps pocket docking scoring \
  --protein-pdb       "6WAU.pdb" \
  --ligand-csv        "/users/s/h/shuhang/PHF19/phf19_data_30mar2026.csv" \
  --gnina-binary      "apptainer run --nv /work/users/s/h/shuhang/gnina_build/gnina.sif" \
  --output-dir        "${OUTPUT_DIR}"
