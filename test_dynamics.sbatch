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

export PIXI_HOME=/work/users/s/h/shuhang/.pixi
export PATH="$PIXI_HOME/bin:$PATH"

cd /work/users/s/h/shuhang/draco

pixi run python test_dynamics.py \
  --protein-pdb        "6pbc-prepared.pdb" \
  --ligand-smiles      "CN(CC1(CC1)c1ccc(F)cc1)C(=O)[C@H](Cc1csc2ccccc12)Nc1cc(ncn1)C(N)=O" \
  --output-dir         "test_dynamics_output" \
  --platform-name      CUDA \
  --cuda-precision     double \
  --box-padding-nm     1.0 \
  --ionic-strength     0.15 \
  --minimize-max-iterations 10000 \
  --nvt-steps          50000  \
  --npt-steps          50000  \
  --production-steps   25000000 \
  --report-interval-steps 5000 \
  --rmsd-threshold-angstrom 1.5 \
  --num-conformers     10 \
  --poses-per-pocket   5 \
  --overlay-dedupe-rmsd 1.0 \
  --pocket-score-threshold 5.0 \
  --shell-radius-angstrom  8.0 \
  --protein-restraint-k    10.0 \
  --ligand-restraint-k     1.0 \
  --equil-restraint-k      2.0 \
  --nvt-timestep-fs        1.0 \
  --minimize-pose-iterations 1000 \
  --top-k 5 \
  --top-pdb "top5_poses.pdb"
