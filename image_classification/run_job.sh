#!/bin/sh
#SBATCH --job-name tinyml-ic_model_ev_assist
#SBATCH --time 30:00:00
#SBATCH --output=output_model_ev_assist.txt
nvidia-smi -q | grep -E '(Name|UUID)'

eval "$(conda shell.bash hook)"
conda activate tiny

python train_resnet.py --isTrecx --isEV --model_architecture=resnet_ev --model_save_name=model_with_ev_assist

