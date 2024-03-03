#!/bin/bash -l

#SBATCH
#SBATCH --time=30:00
#SBATCH --partition=a100
#SBATCH --account=lisik3_gpu
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/home/kgarci18/scratch4-lisik3/SIfMRI_modeling/data/logs/slurm-%j.out

###To submit the job array
#  file="/home/kgarci18/scratch4-lisik3/SIfMRI_modeling/data/raw/model_list/vision_models.csv"; num_models=$(($(wc -l < "$file") - 1)); sbatch --array=1-$num_models%6 batch_vision_encoding_array.sh

###To save unfinished tasks
#  for f in slurm*out; do if ! grep -q "Finished" "$f"; then echo "$(echo $f | sed -n 's/slurm-\([0-9]*\).out/\1/p'),$(grep "model name=" "$f" | sed -n 's/.*model name= \(.*\)/\1/p'),$(tail -n 1 "$f")" >> unfinished_tasks.txt; fi; done

# Path to your CSV file
file="/home/kgarci18/scratch4-lisik3/SIfMRI_modeling/data/raw/model_list/remaining_models.csv"

# SLURM_ARRAY_TASK_ID corresponds to the line number starting from 1
# Adjust by adding 1 to skip the header line
model_line=$((SLURM_ARRAY_TASK_ID + 1))

# Extract the model name from the specified line
model=$(sed -n "${model_line}p" "$file" | cut -d',' -f1)

# Your sbatch command, using the extracted model name
source batch_vision_encoding.sh "$model"

