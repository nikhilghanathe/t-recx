#!/bin/sh
#SBATCH --job-name tinyml-kws_model_noev_assist_point3
#SBATCH --time 30:00:00
#SBATCH --output=output_model_noev_assist_point3.txt
nvidia-smi -q | grep -E '(Name|UUID)'
eval "$(conda shell.bash hook)"
conda activate trecx

python train_dscnn.py --data_dir=data/ --model_architecture=ds_cnn_noev --isTrecx  --model_save_name=model_without_ev_assist_point3 --W-aux=0.3

