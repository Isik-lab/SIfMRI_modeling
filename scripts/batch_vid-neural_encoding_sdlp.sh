#!/bin/bash -l

#SBATCH
#SBATCH --time=6:00:00
#SBATCH --partition=ica100
#SBATCH --account=lisik3_gpu
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%j.out

# Usage:
#   sbatch batch_vid-neural_encoding_sdlp.sh [model] [sdlp_repo] [sdlp_ckpt]
# Defaults target the visual-only Stage-1 checkpoint.
model=${1:-sdlp}
sdlp_repo=${2:-/data/lisik3/kgarci18/sdlp}
sdlp_ckpt=${3:-$sdlp_repo/data/checkpoints/sdlp/stage1.pt}
echo "model name = $model"
echo "sdlp ckpt  = $sdlp_ckpt"

user=$(whoami)
project_folder="/home/$user/scratch4-lisik3/$user/SIfMRI_modeling"

export HF_HOME="${project_folder}/.cache/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="${project_folder}/.cache/huggingface/hub"
export HF_DATASETS_CACHE="${project_folder}/.cache/huggingface/hub"
export TORCH_HOME="${project_folder}/.cache/torch"

# Make the sdlp package importable (alternative: `pip install -e $sdlp_repo` once).
export PYTHONPATH="${sdlp_repo}:${PYTHONPATH:-}"

ml anaconda
conda activate ~/anaconda3/envs/deepjuice

# NOTE: the deepjuice env needs transformers>=4.52 (VJEPA2 support) to load SDLP's
# frozen V-JEPA 2 backbone.  pip install -U "transformers>=4.52" if the run errors
# at model load with a VJEPA2Model ImportError.
~/anaconda3/envs/deepjuice/bin/python video_neural_encoding.py \
  --model_name "$model" \
  --model_input videos \
  --sdlp_ckpt "$sdlp_ckpt" \
  --overwrite \
  --user "$user"
