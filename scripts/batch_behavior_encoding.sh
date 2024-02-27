#!/bin/bash -l

#SBATCH
#SBATCH --time=10:00
#SBATCH --partition=a100
#SBATCH --account=lisik3_gpu
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%j.out

####file=../data/raw/model_list/vision_models.csv; tail -n +2 "$file" | while IFS=',' read -r model cols; do sbatch -J $model batch_behavior_encoding.sh $model; done
####unfinished_output_file="unfinished_tasks.txt"; while IFS= read -r model; do sbatch -J $model batch_behavior_encoding.sh $model; done < "$unfinished_output_file"

model=${1:-torchvision_alexnet_imagenet1k_v1}
echo $model

export HF_HOME='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_modeling/.cache/huggingface/hub'
export HUGGINGFACE_HUB_CACHE='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_modeling/.cache/huggingface/hub'
export HF_DATASETS_CACHE='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_modeling/.cache/huggingface/hub'

ml anaconda
conda activate deepjuice

python vision_behavior_encoding.py --model_uid $model --overwrite
