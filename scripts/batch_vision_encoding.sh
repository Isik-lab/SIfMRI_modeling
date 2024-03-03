#!/bin/bash -l

model=${1:-noModel}
echo "model name= $model"

project_folder="/home/kgarci18/scratch4-lisik3/SIfMRI_modeling"

export HF_HOME="${project_folder}/.cache/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="${project_folder}/.cache/huggingface/hub"
export HF_DATASETS_CACHE="${project_folder}/.cache/huggingface/hub"

ml anaconda
conda activate /home/kgarci18/miniconda3/envs/deepjuice
~/miniconda3/envs/deepjuice/bin/python vision_neural_encoding.py --model_uid $model --overwrite --test_set_evaluation --top_dir $project_folder
