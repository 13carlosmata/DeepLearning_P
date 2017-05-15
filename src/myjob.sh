#!/bin/bash

# The name of the script is myjob
#SBATCH -J myjob

# Time definition
#SBATCH -t 23:50:00

# set the project to be charged for this
#SBATCH -A edu17.DD2424

# Use K80 GPUs
#SBATCH --gres=gpu:K80:2

# Standard error and standard output to files
#SBATCH -e error_file.txt
#SBATCH -o output_file.txt

# Run the executable
module add cudnn/5.1-cuda-8.0
module load anaconda/py27/4.2.0
python2 src/main.py
