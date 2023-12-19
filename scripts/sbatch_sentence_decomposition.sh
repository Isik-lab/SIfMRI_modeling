#!/bin/bash -l

#SBATCH
#SBATCH --job-name=sentence_decomp
#SBATCH --time=50:00
#SBATCH --partition=a100
#SBATCH --account=lisik3_gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end
#SBATCH --mail-user=emcmaho7@jh.edu

func=${1:-none}

ml anaconda
conda activate fmri_modeling

python sentence_decomposition.py --func_name $func --overwrite
