#!/bin/sh
#SBATCH --job-name tinyml-model_with_ev_assist
#SBATCH --time 30:00:00
#SBATCH --output=output_model_with_ev_assist.txt
nvidia-smi -q | grep -E '(Name|UUID)'
eval "$(conda shell.bash hook)"
conda activate tiny
model_name='trained_models/model_with_ev_assist'
model_architecture='mobnet_ev'
python train_mobnet.py $model_name $model_architecture

