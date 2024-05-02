#!/bin/bash -l

model=${1:-slowfast_r50}
echo "model name= $model"

user=$(whoami)
project_folder="/home/$user/scratch4-lisik3/$user/SIfMRI_modeling"

export HF_HOME="${project_folder}/.cache/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="${project_folder}/.cache/huggingface/hub"
export HF_DATASETS_CACHE="${project_folder}/.cache/huggingface/hub"
export TORCH_HOME="${project_folder}/.cache/torch"

ml anaconda
conda activate ~/miniconda3/envs/deepjuice

~/miniconda3/envs/deepjuice/bin/python video_neural_encoding.py --model_name $model --model_input videos --overwrite --user $user