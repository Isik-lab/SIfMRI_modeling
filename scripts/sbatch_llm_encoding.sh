#!/bin/bash -l

#SBATCH
#SBATCH --job-name=model_encoding
#SBATCH --time=10:00
#SBATCH --partition=a100
#SBATCH --account=lisik3_gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end
#SBATCH --mail-user=emcmaho7@jh.edu

perturbation=${1:-stripped_orig}
model=${2:-sentence-transformers/all-MiniLM-L6-v2}

echo "perturbation: $perturbation"
echo "model: $model"

ml anaconda
conda activate deepjuice

python llm_encoding.py --model_uid $model --perturbation $perturbation --overwrite
