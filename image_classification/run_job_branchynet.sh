#!/bin/sh
#SBATCH --job-name tinyml-ic_model_branchynet
#SBATCH --time 30:00:00
#SBATCH --output=output_model_branchynet.txt
nvidia-smi -q | grep -E '(Name|UUID)'

eval "$(conda shell.bash hook)"
conda activate tiny

python train_resnet.py --model_architecture=resnet_branchynet --model_save_name=model_branchynet

