#!/bin/bash -l

model=${1:-sentence-transformers/all-MiniLM-L6-v2}
echo "model name= $model"

user=$(whoami)
project_folder="/home/$user/scratch4-lisik3/$user/SIfMRI_modeling"

export HF_HOME="${project_folder}/.cache/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="${project_folder}/.cache/huggingface/hub"
export HF_HUB_CACHE="${project_folder}/.cache/huggingface/hub"
export HF_DATASETS_CACHE="${project_folder}/.cache/huggingface/hub"

ml anaconda
conda activate deepjuice

python language_neural_encoding.py --model_uid $model \
    --overwrite --test_eval \
    --top_dir $project_folder \
    --user $user
