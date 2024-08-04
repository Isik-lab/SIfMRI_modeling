#!/bin/bash -l

#SBATCH
#SBATCH --time=2:30:00
#SBATCH --partition=v100
#SBATCH --account=lisik3_gpu
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%A_%a.out

model=${1:-dorsalnet}
grouping=${2:-first_frame}
echo "model name= $model"

user=$(whoami)
project_folder="/home/$user/scratch4-lisik3/$user/SIfMRI_modeling"

export HF_HOME="${project_folder}/.cache/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="${project_folder}/.cache/huggingface/hub"
export HF_HUB_CACHE="${project_folder}/.cache/huggingface/hub"
export HF_DATASETS_CACHE="${project_folder}/.cache/huggingface/hub"

ml anaconda
conda activate ~/anaconda3/envs/deepjuice

~/anaconda3/envs/deepjuice/bin/python vision_neural_encoding.py --model_uid $model \
    --test_eval --overwrite \
    --top_dir $project_folder \
    --frame_handling $grouping \
    --user $user
