5#!/bin/sh
#SBATCH --job-name tinyml-model_with_ev_assist_point5
#SBATCH --time 30:00:00
#SBATCH --output=output_model_with_ev_assist_point5.txt
nvidia-smi -q | grep -E '(Name|UUID)'
eval "$(conda shell.bash hook)"
conda activate trecx
model_name='model_with_ev_assist_point5'
model_architecture='mobnet_ev'
python train_mobnet.py $model_name $model_architecture

