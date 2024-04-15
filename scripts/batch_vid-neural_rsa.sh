#!/bin/bash -l

#SBATCH
#SBATCH --time=2:00:00
#SBATCH --partition=ica100
#SBATCH --account=lisik3_gpu
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%j.out

model=${1:-slip_vit_s_yfcc15m}

user=$(whoami)
project_folder="/home/$user/scratch4-lisik3/$user/SIfMRI_modeling"

export HF_HOME="${project_folder}/.cache/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="${project_folder}/.cache/huggingface/hub"
export HF_HUB_CACHE="${project_folder}/.cache/huggingface/hub"
export HF_DATASETS_CACHE="${project_folder}/.cache/huggingface/hub"

ml anaconda
conda activate /home/kgarci18/miniconda3/envs/deepjuice

~/miniconda3/envs/deepjuice/bin/python video_neural_rsa.py --model_uid $model --data_dir "$project_folder/data" --overwrite --user $user
