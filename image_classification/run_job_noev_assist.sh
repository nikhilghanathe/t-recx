#!/bin/sh
#SBATCH --job-name tinyml-ic_model_noev_assist
#SBATCH --time 30:00:00
#SBATCH --output=output_model_noev_assist.txt
nvidia-smi -q | grep -E '(Name|UUID)'

eval "$(conda shell.bash hook)"
conda activate trecx

python train_resnet.py --isTrecx --model_architecture=resnet_noev --model_save_name=model_without_ev_assist

