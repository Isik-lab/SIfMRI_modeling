#!/bin/bash -l

#SBATCH
#SBATCH --time=30:00
#SBATCH --partition=a100
#SBATCH --account=lisik3_gpu
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%j.out

model=${1:-slip_vit_s_yfcc15m}


ml anaconda
conda activate deepjuice

python rsa_benchmark.py --model_uid $model --data_dir "/home/kgarci18/scratch4-lisik3/kgarci18/SIfMRI_modeling/data" --overwrite
