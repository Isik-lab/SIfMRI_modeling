#!/bin/bash -l

#SBATCH
#SBATCH --job-name=slip
#SBATCH --time=10:00
#SBATCH --partition=a100
#SBATCH --account=lisik3_gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end
#SBATCH --mail-user=emcmaho7@jh.edu
#SBATCH --output=slurm-%x-%j.out

backbone=${1:-clip_small_25ep}
perturbation=${2:-stripped_orig}

echo "perturbation: $perturbation"
echo "backbone: $backbone"

ml anaconda
conda activate slip

python slip_llm_encoding.py --backbone $backbone --perturbation $perturbation --overwrite
