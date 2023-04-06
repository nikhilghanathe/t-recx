#!/bin/sh
#SBATCH --job-name tinyml-kws_test
#SBATCH --time 30:00:00
#SBATCH --output=output_test.txt
nvidia-smi -q | grep -E '(Name|UUID)'

eval "$(conda shell.bash hook)"
conda activate trecx

python test_dscnn.py

