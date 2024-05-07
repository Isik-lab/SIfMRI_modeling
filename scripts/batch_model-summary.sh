#!/bin/bash -l

#SBATCH
#SBATCH --time=10:00:00
#SBATCH --partition=shared
#SBATCH --account=lisik33
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm-%A.out

model_class="VisionNeuralEncoding"
model_subpath=""
category_col=""

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_class)
            model_class="$2"
            shift # remove the argument name
            shift # remove the argument value
            ;;
        --model_subpath)
            model_subpath="$2"
            shift # remove the argument name
            shift # remove the argument value
            ;;
        --category_col)
            category_col="$2"
            shift # remove the argument name
            shift # remove the argument value
            ;;
        *)    # unknown option
            shift # skip unknown option
            ;;
    esac
done

echo "Model Class: $model_class"
echo "Model Subpath: $model_subpath"
echo "Category Column: $category_col"

ml anaconda 
conda activate deepjuice

command="python model_summary.py --model_class $model_class"
if [[ -n $model_subpath ]]; then
  command+=" --model_subpath $model_subpath"
fi

if [[ -n $category_col ]]; then
  command+=" --category_col $category_col"
fi

echo $command
# Execute the command
eval $command
