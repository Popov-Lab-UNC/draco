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
#SBATCH --output=test_1_%j.log
#SBATCH --error=test_1_%j.err
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
  --mode single \
  --steps docking scoring \
  --protein-pdb    "4GL2.pdb" \
  --ligand-smiles  "Cc1ccc(S(N)(=O)=O)cc1C(=O)NCc1ccc(-c2nc3ccccc3s2)o1" \
  --gnina-binary   "apptainer run --nv /work/users/s/h/shuhang/gnina_build/gnina.sif" \
  --output-dir     "4GL2_XS421854b" \
  --num-conformers 100
'''
pixi run draco \
  --mode single \
  --steps docking scoring \
  --protein-pdb    "4GL2.pdb" \
  --ligand-smiles  "O=C(Nc1ccc(Cl)cc1)Nc1ccc(S(=O)(=O)Nc2nccs2)cc1" \
  --gnina-binary   "apptainer run --nv /work/users/s/h/shuhang/gnina_build/gnina.sif" \
  --output-dir     "4GL2_XS466077b" \
  --num-conformers 100
'''