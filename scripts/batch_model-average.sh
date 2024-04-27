#!/bin/bash -l

#SBATCH
#SBATCH --time=10:00:00
#SBATCH --partition=shared
#SBATCH --account=lisik33
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm-%A.out

model_class=${1:-"VisionNeuralEncoding"}
model_subpath=${2:-grouped_average}

ml anaconda 
conda activate deepjuice

# Your sbatch command, using the extracted model name
python model_averaging.py --model_class $model_class --model_subpath $model_subpath
