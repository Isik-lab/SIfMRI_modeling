#!/bin/bash -l

#SBATCH --time=4:00:00
#SBATCH --partition=ica100
#SBATCH --account=lisik3_gpu
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --output=slurm-%A_%a.out

# Parameters
file="../data/raw/model_list/language_models.csv"
funcs_file="../data/raw/function_list/perturbations.csv"

# Read function names from CSV, skipping the header
mapfile -t funcs < <(tail -n +2 "$funcs_file")

num_funcs=${#funcs[@]}
num_models=$(($(wc -l < "$file") - 1))  # Subtract 1 for the header

# Calculate total number of tasks
total_tasks=$((num_models * num_funcs))

# Determine the model and function for the current task
model_index=$(( (SLURM_ARRAY_TASK_ID - 1) % num_models + 2 ))  # +2 to skip header and adjust index
func_index=$(( (SLURM_ARRAY_TASK_ID - 1) / num_models ))
model=$(sed -n "${model_index}p" "$file" | cut -d',' -f1)
func=${funcs[$func_index]}
echo "function = $func"

# Execute the task
source batch_lm-beh_encoding.sh "$model" "$func"