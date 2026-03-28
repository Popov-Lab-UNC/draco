#!/bin/bash
#SBATCH --job-name=draco_dynamics

#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1

##SBATCH --partition=general
#SBATCH -n 1
#SBATCH --cpus-per-task=16

#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=test_dynamics_%j.log
#SBATCH --error=test_dynamics_%j.err
#SBATCH --mail-user=shuhang@unc.edu
#SBATCH --mail-type=ALL

module load cuda
module load apptainer

export PIXI_HOME=/work/users/s/h/shuhang/.pixi
export PATH="$PIXI_HOME/bin:$PATH"

cd /work/users/s/h/shuhang/draco

pixi run python test_dynamics.py \
  --protein-pdb        "6WAU.pdb" \
  --ligand-smiles      "BrC1=NN(C)C2=C1C(N3CC[C@@H](NC4CCC4)CCC3)=NC(N5CCC(F)(F)C5)=N2" \
  --gnina-binary       "apptainer run --nv /work/users/s/h/shuhang/gnina_build/gnina.sif" \
  --output-dir         "test_dynamics_output_2" \
  --platform-name      CUDA \
  --cuda-precision     mixed \
  --box-padding-nm     1.0 \
  --ionic-strength     0.15 \
  --water-model        tip3pfb \
  --nvt-steps          50000 \
  --npt-steps          50000 \
  --production-steps   25000000 \
  --report-interval-steps 5000 \
  --rmsd-threshold-angstrom 1.5 \
  --num-conformers     10 \
  --num-modes          5 \
  --pocket-score-threshold 5.0 \
  --shell-radius-angstrom  8.0 \
  --protein-restraint-k    10.0 \
  --refine-iterations      500 \
  --top-k 5 \
  --top-pdb "top5_poses.pdb"
