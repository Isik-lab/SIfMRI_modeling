#!/bin/bash -l

#SBATCH
#SBATCH --job-name=testing
#SBATCH --time=45:00
#SBATCH --partition=a100
#SBATCH --account=lisik3_gpu
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-gpu=10
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --mail-type=end
#SBATCH --mail-user=emcmaho7@jh.edu

perturbation=${1:-none}
model=${1:-sentence-transformers/all-MiniLM-L6-v2}

echo "perturbation: $perturbation"
echo "model: $model"

ml anaconda
conda activate fmri_modeling

python llm_encoding.py --model_uid $model --perturbation $perturbation --overwrite
