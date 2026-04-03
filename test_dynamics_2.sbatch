#!/bin/bash
#SBATCH --job-name=draco_dynamics

#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1

##SBATCH --partition=general
#SBATCH -n 1
#SBATCH --cpus-per-task=16

#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=test_2_%j.log
#SBATCH --error=test_2_%j.err
#SBATCH --mail-user=shuhang@unc.edu
#SBATCH --mail-type=ALL

module load cuda
module load apptainer

export PIXI_HOME=/work/users/s/h/shuhang/.pixi
export PATH="$PIXI_HOME/bin:$PATH"

cd /work/users/s/h/shuhang/draco

OUTPUT_DIR="6WAU-frame7-3apr2026"

pixi run draco \
  --mode single \
  --steps dynamics pocket docking scoring refinement \
  --dock-filter -1 \
  --protein-pdb       "6WAU-SAR-30mar2026/frames/frame_0007.pdb" \
  --ligand-smiles        "Cn1nc(Br)c2c(N3CCC[C@@H](NC4CCC4)C3)nc(N3CCC(F)(F)C3)nc21" \
  --gnina-binary      "apptainer run --nv /work/users/s/h/shuhang/gnina_build/gnina.sif" \
  --output-dir        "${OUTPUT_DIR}" \
  --production-steps 500000 \
  --rmsd-threshold-angstrom 0.5 \
  --gnina-timeout-seconds 180 \
  --refine-iterations 10000
