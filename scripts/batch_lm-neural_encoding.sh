#!/bin/bash -l

model=${1:-sentence-transformers/all-MiniLM-L6-v2}
func=${2:-mask_nouns}
echo "model name = $model"
echo "function = $func"

user=$(whoami)
echo "user = $user"
project_folder="/home/$user/scratch4-lisik3/$user/SIfMRI_modeling"

export HF_HOME="${project_folder}/.cache/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="${project_folder}/.cache/huggingface/hub"
export HF_HUB_CACHE="${project_folder}/.cache/huggingface/hub"
export HF_DATASETS_CACHE="${project_folder}/.cache/huggingface/hub"

ml anaconda
conda activate deepjuice

echo "python language_behavior_encoding.py --model_uid $model --overwrite \
    --top_dir $project_folder --user $user --perturb_func $func" 

python language_neural_encoding.py --model_uid $model \
    --test_eval --perturb_func $func \
    --top_dir $project_folder \
    --user $user --stream_statistics
