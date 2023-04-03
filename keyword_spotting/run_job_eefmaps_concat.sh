-#!/bin/sh
#SBATCH --job-name tinyml-kws_model_eefmaps_concat
#SBATCH --time 30:00:00
#SBATCH --output=output_model_eefmaps_concat.txt
nvidia-smi -q | grep -E '(Name|UUID)'
eval "$(conda shell.bash hook)"
conda activate tiny

python train_dscnn.py --data=data/ --model_architecture=ds_cnn_eefmaps_concat --isTrecx --model_save_name=model_eefmaps_concat

