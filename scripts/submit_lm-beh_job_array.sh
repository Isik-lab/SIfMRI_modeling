#!/bin/bash

# Parameters
model_file="../data/raw/model_list/language_models.csv"
func_file="../data/raw/model_list/perturbations.csv"
num_funcs=$(($(wc -l < "$func_file") - 1))  # Subtract 1 for the header
num_models=$(($(wc -l < "$model_file") - 1))  # Subtract 1 for the header
echo "number of functions: $num_funcs" 
echo "number of models: $num_models" 

# Calculate total number of tasks
total_tasks=$((num_models * num_funcs))
echo "total number of tasks: " "$total_tasks" 

# Submit the job array with the calculated total tasks
sbatch --array=1-$total_tasks%4 batch_lm-beh_encoding_array.sh