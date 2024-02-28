#!/bin/bash -l

model=${1:-torchvision_alexnet_imagenet1k_v1}
echo "model name= $model"

export HF_HOME='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_modeling/.cache/huggingface/hub'
export HUGGINGFACE_HUB_CACHE='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_modeling/.cache/huggingface/hub'
export HF_DATASETS_CACHE='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_modeling/.cache/huggingface/hub'

ml anaconda
conda activate deepjuice

python vision_behavior_encoding.py --model_uid $model --overwrite
