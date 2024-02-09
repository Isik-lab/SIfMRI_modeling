#!/bin/bash -l

#SBATCH
#SBATCH --time=10:00
#SBATCH --partition=a100
#SBATCH --account=lisik3_gpu
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-bert-%x-%j.out

perturbation=${1:-stripped_orig}
model=${2:-sentence-transformers/all-MiniLM-L6-v2}

echo "perturbation: $perturbation"
echo "model: $model"

ml anaconda
conda activate deepjuice_stable

python llm_encoding.py --model_uid $model --perturbation $perturbation --overwrite
