#!/bin/bash -l

#SBATCH --time=4:00:00
#SBATCH --partition=ica100
#SBATCH --account=lisik3_gpu
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --output=slurm-%A_%a.out

model=${1:-xclip-base-patch32}
echo "model name= $model"

user=$(whoami)
project_folder="/home/$user/scratch4-lisik3/$user/SIfMRI_modeling"

export HF_HOME="${project_folder}/.cache/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="${project_folder}/.cache/huggingface/hub"
export HF_DATASETS_CACHE="${project_folder}/.cache/huggingface/hub"
export TORCH_HOME="${project_folder}/.cache/torch"

ml anaconda
conda activate ~/miniconda3/envs/deepjuice

~/miniconda3/envs/deepjuice/bin/python multimodal_beh_encoding.py --model_name $model --model_input videos --overwrite --user $user
