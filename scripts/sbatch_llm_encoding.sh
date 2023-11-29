#!/bin/bash -l

#SBATCH
#SBATCH --job-name=model_encoding
#SBATCH --time=45:00
#SBATCH --partition=a100
#SBATCH --account=lisik3_gpu
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end
#SBATCH --mail-user=emcmaho7@jh.edu

perturbation=$1

ml anaconda
conda activate fmri_modeling

python llm_encoding.py --perturbation $perturbation --overwrite
