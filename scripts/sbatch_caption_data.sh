#!/bin/bash -l

#SBATCH
#SBATCH --acount=lisik3
#SBATCH --job-name=CaptionData
#SBATCH --time=10:00
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=end
#SBATCH --mail-user=emcmaho7@jhu.edu

ml anaconda
conda activate llm

python caption_data.py
