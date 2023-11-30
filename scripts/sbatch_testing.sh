#!/bin/bash -l

#SBATCH
#SBATCH --job-name=testing
#SBATCH --time=15:00
#SBATCH --partition=a100
#SBATCH --account=lisik3_gpu
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-gpu=10
#SBATCH --gpus-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --mail-type=end
#SBATCH --mail-user=emcmaho7@jh.edu

perturbation=$1

ml anaconda
conda activate fmri_modeling

python testing.py
