#!/bin/bash -l

#SBATCH
#SBATCH --job-name=testing
#SBATCH --time=5:00
#SBATCH --partition=a100
#SBATCH --account=lisik3_gpu
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=5G
#SBATCH --cpus-per-gpu=5
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --mail-type=end
#SBATCH --mail-user=emcmaho7@jh.edu

perturbation=${1:-corrected_captions}
model=${2:-sentence-transformers/all-MiniLM-L6-v2}

echo "perturbation: $perturbation"
echo "model: $model"

ml anaconda
conda activate fmri_modeling

python llm_encoding.py --model_uid $model --perturbation $perturbation --overwrite
