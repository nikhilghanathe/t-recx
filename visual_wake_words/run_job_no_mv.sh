#!/bin/sh
#SBATCH --job-name tinyml-vww_model_no_mv_ee_at_quarter_with_weighted_loss_point3
#SBATCH --time 40:00:00
#SBATCH --output=output_model_no_mv_ee_at_quarter_with_weighted_loss_point3.txt
nvidia-smi -q | grep -E '(Name|UUID)'
eval "$(conda shell.bash hook)"
conda activate tiny
model_name='trained_models/model_no_mv_ee_at_quarter_with_weighted_loss_point3'

python3 train_vww_no_mv.py $model_name
python3 test_vww.py $model_name
# python test_sep_loss_conv_plus_dense.py trained_models/test_on_batch_begin_ch32_no_neg
#python3 train_orig_ee1_plus_eefinal.py 0
#python3 test_orig_plus_ee1.py

