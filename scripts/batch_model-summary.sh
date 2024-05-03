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
model_subpath=$2

ml anaconda 
conda activate deepjuice

command="python model_summary.py --model_class $model_class"
if [[ -n $model_subpath ]]; then
  command+=" --model_subpath $model_subpath"
fi

echo $command
# Execute the command
# eval $command
