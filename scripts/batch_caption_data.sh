#!/bin/bash -l

#SBATCH
#SBATCH --job-name=glmsingle
#SBATCH --time=15:00
#SBATCH --partition=a100
#SBATCH --account=lisik3_gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end
#SBATCH --mail-user=emcmaho7@jh.edu

ml anaconda
conda activate deepjuice

python caption_data.py
