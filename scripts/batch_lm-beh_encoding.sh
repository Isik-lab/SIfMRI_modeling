#!/bin/bash -l

model=${1:-torchvision_alexnet_imagenet1k_v1}
func=${1:-mask_nouns}
echo "model name= $model"

user=$(whoami)
project_folder="/home/$user/scratch4-lisik3/$user/SIfMRI_modeling"

export HF_HOME="${project_folder}/.cache/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="${project_folder}/.cache/huggingface/hub"
export HF_HUB_CACHE="${project_folder}/.cache/huggingface/hub"
export HF_DATASETS_CACHE="${project_folder}/.cache/huggingface/hub"

ml anaconda
conda activate deepjuice

python language_behavior_encoding.py --model_uid $model --overwrite \
    --top_dir $project_folder --user $user --perturbation \
    --perturb_func $func
