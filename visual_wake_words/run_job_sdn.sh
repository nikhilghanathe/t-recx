#!/bin/sh
#SBATCH --job-name tinyml-vww_model_sdn_no_lweight
#SBATCH --time 40:00:00
#SBATCH --output=output_model_sdn_no_lweight.txt
nvidia-smi -q | grep -E '(Name|UUID)'
eval "$(conda shell.bash hook)"
conda activate tiny
model_name='trained_models/model_sdn_no_lweight'
python train_vww_prior.py $model_name 'sdn'
python test_vww.py $model_name

