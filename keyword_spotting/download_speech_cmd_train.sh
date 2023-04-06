#!/bin/bash
export TF_CPP_MIN_LOG_LEVEL=2 
# train all models for testing
#model with EV-assistance
echo '======================================================'
echo "Training T-trecx Model with EV-assistance....."
echo '======================================================'
sleep 2
python train_dscnn.py --data_dir=data/ --model_architecture='ds_cnn_ev' --isTrecx --isEV --model_save_name='model_with_ev_assist'
#model without EV-assistance
echo '======================================================'
echo "Training T-trecx Model without EV-assistance....."
echo '======================================================'
sleep 2
python train_dscnn.py --data_dir=data/ --model_architecture='ds_cnn_noev' --isTrecx --model_save_name='model_without_ev_assist'
#model EE-fmaps concat
echo '======================================================'
echo "Training DSCNN Model with EE-fmaps concatenated with final-fmaps....."
echo '======================================================'
sleep 2
python train_dscnn.py --data_dir=data/ --model_architecture='ds_cnn_eefmaps_concat' --isTrecx --model_save_name='model_eefmaps_concat'

#DSCNN with SDN techniques
echo '======================================================'
echo "Training SDN-DSCNN....."
echo '======================================================'
sleep 2
python train_dscnn.py --data_dir=data/ --model_architecture='ds_cnn_sdn' --model_save_name='model_sdn'
#DSCNN with Branchynet techniques
echo '======================================================'
echo "Training Branchynet-DSCNN....."
echo '======================================================'
sleep 2
python train_dscnn.py --data_dir=data/ --model_architecture='ds_cnn_branchynet' --model_save_name='model_branchynet'