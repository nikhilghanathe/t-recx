#!/bin/sh
#SBATCH --job-name tinyml-kws_model_branchynet
#SBATCH --time 30:00:00
#SBATCH --output=output_model_branchynet.txt
nvidia-smi -q | grep -E '(Name|UUID)'
eval "$(conda shell.bash hook)"
conda activate trecx

python train_dscnn.py --data_dir=data/ --plot_dir=plots/  --model_architecture=ds_cnn_branchynet --model_save_name=model_branchynet

