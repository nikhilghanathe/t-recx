#!/bin/sh
#SBATCH --job-name tinyml-ic_test_on_batch_begin_ch64
#SBATCH --time 30:00:00
#SBATCH --output=output_test_on_batch_begin_ch64.txt
nvidia-smi -q | grep -E '(Name|UUID)'
eval "$(conda shell.bash hook)"
conda activate tiny
model_name='trained_models/model_ee_at_quarter_with_weighted_loss_point2'
python train_vww_test.py --model_save_name=$model_name
python test_vww	.py $model_name
# python test_sep_loss_conv_plus_dense.py trained_models/test_on_batch_begin_ch32_no_neg
#python3 train_orig_ee1_plus_eefinal.py 0
#python3 test_orig_plus_ee1.py

