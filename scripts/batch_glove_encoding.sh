#!/bin/bash -l

#SBATCH
#SBATCH --time=10:00
#SBATCH --partition=a100
#SBATCH --account=lisik3_gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-glove-%x-%j.out

perturbation=${1:-stripped_orig}
echo "perturbation: $perturbation"

ml anaconda
conda activate deepjuice

python glove_encoding.py --perturbation $perturbation --overwrite
