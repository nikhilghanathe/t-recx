#!/bin/sh
#SBATCH --job-name tinyml-kws_model_ev_assist
#SBATCH --time 30:00:00
#SBATCH --output=output_model_ev_assist.txt
nvidia-smi -q | grep -E '(Name|UUID)'
eval "$(conda shell.bash hook)"
conda activate trecx

python train_dscnn.py --data_dir=data/ --model_architecture=ds_cnn_ev --isTrecx --isEV --model_save_name=model_with_ev_assist

