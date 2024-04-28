#!/bin/bash

# Parameters
file="../data/raw/model_list/language_models.csv"
funcs=(none shuffle mask_nouns mask_verbs mask_nonnouns mask_nonverbs)
num_funcs=${#funcs[@]}
num_models=$(($(wc -l < "$file") - 1))  # Subtract 1 for the header
echo "number of functions: $num_funcs" 
echo "number of models: $num_models" 

# Calculate total number of tasks
total_tasks=$((num_models * num_funcs))
echo "total number of tasks: " "$total_tasks" 

# Submit the job array with the calculated total tasks
sbatch --array=1-$total_tasks%6 batch_lm-beh_encoding_array.sh