#!/bin/bash -l

#SBATCH
#SBATCH --time=30:00
#SBATCH --partition=a100
#SBATCH --account=lisik3_gpu
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --output=slurm-%x-%j.out

model=${1:-slip_vit_s_yfcc15m}

user=$(whoami)
project_folder="/home/$user/scratch4-lisik3/$user/SIfMRI_modeling"

ml anaconda
conda activate /home/kgarci18/miniconda3/envs/deepjuice
export HF_HOME="/home/kgarci18/scratch4-lisik3/SIfMRI_modeling/.cache"
~/miniconda3/envs/deepjuice/bin/python rsa_benchmark.py --model_uid $model --data_dir "$project_folder/data" --overwrite  --user $user
