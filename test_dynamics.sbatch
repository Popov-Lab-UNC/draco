#!/bin/bash
#SBATCH --job-name=draco_dynamics

#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1

##SBATCH --partition=general
##SBATCH -n 1
#SBATCH --cpus-per-task=16

#SBATCH --mem=32G
#SBATCH --time=1-12:00:00
#SBATCH --output=test_dynamics_%j.log
#SBATCH --error=test_dynamics_%j.err
#SBATCH --mail-user=shuhang@unc.edu
#SBATCH --mail-type=ALL

module load cuda
module load apptainer

export PIXI_HOME=/work/users/s/h/shuhang/.pixi
export PATH="$PIXI_HOME/bin:$PATH"

cd /work/users/s/h/shuhang/draco

# 6pbc-prepared.pdb
# CN(CC1(CC1)c1ccc(F)cc1)C(=O)[C@H](Cc1csc2ccccc12)Nc1cc(ncn1)C(N)=O

pixi run draco \
  --protein-pdb        "4GL2.pdb" \
  --ligand-smiles      "O=C(Nc1ccc(Cl)cc1)Nc1ccc(S(=O)(=O)Nc2nccs2)cc1" \
  --gnina-binary       "apptainer run --nv /work/users/s/h/shuhang/gnina_build/gnina.sif" \
  --output-dir         "test_dynamics_output" \
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
  --skip-refinement \
  --top-k 5 \
  --top-pdb "top5_poses.cif" \
  --analyze-frames-only
