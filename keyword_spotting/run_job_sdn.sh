#!/bin/sh
#SBATCH --job-name tinyml-kws_model_sdn
#SBATCH --time 30:00:00
#SBATCH --output=output_model_sdn.txt
nvidia-smi -q | grep -E '(Name|UUID)'
eval "$(conda shell.bash hook)"
conda activate trecx

python train_dscnn.py --data_dir=data/ --plot_dir=plots/  --model_architecture=ds_cnn_sdn --model_save_name=model_sdn

