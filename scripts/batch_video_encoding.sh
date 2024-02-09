#!/bin/bash -l

#SBATCH
#SBATCH --time=2:00:00
#SBATCH --partition=a100
#SBATCH --account=lisik3_gpu
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%j.out

model=${1:-slowfast_r50}

ml anaconda
conda activate deepjuice_video

python video_encoding.py --model_name $model --model_input videos --overwrite
