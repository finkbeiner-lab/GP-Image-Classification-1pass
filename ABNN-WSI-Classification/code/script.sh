#!/bin/bash

#SBATCH --job-name=test      ## Name of the job
#SBATCH --output=test.out    ## Output file
#SBATCH --time=1:00           ## Job Duration
#SBATCH --ntasks=1             ## Number of tasks (analyses) to run
#SBATCH --cpus-per-task=1      ## The number of threads the code will use
#SBATCH --mem-per-cpu=100M     ## Real memory(MB) per CPU required by the job.

## Load the python interpreter

source /gladstone/finkbeiner/home/mahirwar/miniforge3/etc/profile.d/conda.sh
conda activate abnn
module load cuda/12.4
echo "test"

