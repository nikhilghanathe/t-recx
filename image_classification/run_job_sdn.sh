#!/bin/sh
#SBATCH --job-name tinyml-ic_model_sdn
#SBATCH --time 30:00:00
#SBATCH --output=output_model_sdn.txt
nvidia-smi -q | grep -E '(Name|UUID)'

eval "$(conda shell.bash hook)"
conda activate trecx

python train_resnet.py --model_architecture=resnet_sdn --model_save_name=model_sdn

