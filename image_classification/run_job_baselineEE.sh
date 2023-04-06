#!/bin/sh
#SBATCH --job-name tinyml-ic_model_baselineEE
#SBATCH --time 30:00:00
#SBATCH --output=output_model_baselineEE.txt
nvidia-smi -q | grep -E '(Name|UUID)'

eval "$(conda shell.bash hook)"
conda activate trecx

python train_resnet.py --model_architecture=resnet_baselineEE --model_save_name=model_baselineEE

