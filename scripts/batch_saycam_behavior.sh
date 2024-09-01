#!/bin/bash -l

#SBATCH --time=2:00:00
#SBATCH --partition=ica100
#SBATCH --account=lisik3_gpu
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --output=slurm-%A_%a.out

model=${1:-dino_say_vitb14}
echo "model name= $model"

user=$(whoami)
project_folder="/home/$user/scratch4-lisik3/$user/SIfMRI_modeling"

export HF_HOME="${project_folder}/.cache/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="${project_folder}/.cache/huggingface/hub"
export HF_DATASETS_CACHE="${project_folder}/.cache/huggingface/hub"
export TORCH_HOME="${project_folder}/.cache/torch"

ml anaconda
conda activate deepjuice

python saycam_neural_encoding.py --model_uid $model --overwrite --user $user --test_eval --overwrite