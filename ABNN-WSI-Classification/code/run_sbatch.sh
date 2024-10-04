#!/bin/bash

#SBATCH --job-name=test      ## Name of the job
#SBATCH --output=test.out    ## Output file
#SBATCH --time=10:00           ## Job Duration
#SBATCH --ntasks=1             ## Number of tasks (analyses) to run
#SBATCH --cpus-per-task=1      ## The number of threads the code will use
#SBATCH --mem-per-cpu=100M     ## Real memory(MB) per CPU required by the job.

## Load the python interpreters

source /gladstone/finkbeiner/home/mahirwar/miniforge3/etc/profile.d/conda.sh
conda activate abnn
module load cuda/12.4

cd /gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/ABNN-WSI-Classification/code


#python3 ABNN_WSI.py --mode TENSOR --ext svs --model_pretrained True --model_type RESNET18


#python3 ABNN_WSI.py --mode TRAIN --ext pth --num_epoch 50 --batch_size 2

python3 ABNN_WSI.py --mode TEST --ext pth