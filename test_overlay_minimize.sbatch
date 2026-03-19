#!/bin/bash
#SBATCH --job-name=test_overlay_minimize

##SBATCH --partition=volta-gpu
##SBATCH --qos=gpu_access
##SBATCH --gres=gpu:1

#SBATCH --partition=general
#SBATCH -n 1
#SBATCH --cpus-per-task=32

#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=test_overlay_minimize_%j.log
#SBATCH --error=test_overlay_minimize_%j.err
#SBATCH --mail-user=shuhang@unc.edu
#SBATCH --mail-type=ALL

module load cuda

export PIXI_HOME=/work/users/s/h/shuhang/.pixi
export PATH="$PIXI_HOME/bin:$PATH"

cd /work/users/s/h/shuhang/draco

pixi run python test_overlay_minimize.py \
  --protein-pdb "6pbc-prepared.pdb" \
  --ligand-smiles "CN(CC1(CC1)c1ccc(F)cc1)C(=O)[C@H](Cc1csc2ccccc12)Nc1cc(ncn1)C(N)=O" \
  --poses-per-pocket 5 \
  --overlay-dedupe-rmsd 1.0 \
  --output-dir "test_output"

