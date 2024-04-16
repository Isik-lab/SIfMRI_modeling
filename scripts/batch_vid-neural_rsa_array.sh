#SBATCH
#SBATCH --time=6:00:00
#SBATCH --partition=ica100
#SBATCH --account=lisik3_gpu
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%j.out

user=$(whoami)
project_folder="/home/$user/scratch4-lisik3/$user/SIfMRI_modeling"

file=${1:-"$project_folder/data/raw/model_list/video_models.csv"}

###To submit the job array
#  file="/home/kgarci18/scratch4-lisik3/kgarci18/SIfMRI_modeling/data/raw/model_list/video_models.csv"; num_models=$(($(wc -l < "$file") - 1)); sbatch --array=1-$num_models%5 batch_vid-neural_rsa_array.sh $file
#  file="../data/raw/model_list/video_models.csv"; num_models=10; sbatch --array=1-$num_models%2 batch_vid-neural_rsa_array.sh $file

###To save unfinished tasks
# for f in slurm*out; do if ! grep -q "Finished" "$f" && grep -q "VisionNeuralEncoding" "$f"; then echo "$(grep "model name=" "$f" | sed -n 's/.*model name= \(.*\)/\1/p'),$(echo $f | sed -n 's/slurm-\([0-9]*\).out/\1/p'),$(tail -n 1 "$f")" >> unfinished_neural_tasks.txt; fi; done
# find . -name 'slurm*out' -exec grep -ql 'VisionNeuralEncoding' {} \; -delete

# SLURM_ARRAY_TASK_ID corresponds to the line number starting from 1
# Adjust by adding 1 to skip the header line
model_line=$((SLURM_ARRAY_TASK_ID + 1))

# Extract the model name from the specified line
model=$(sed -n "${model_line}p" "$file" | cut -d',' -f1)

# Your sbatch command, using the extracted model name
source batch_vid-neural_rsa.sh "$model"