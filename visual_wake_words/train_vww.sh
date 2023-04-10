#!/bin/bash

export TF_CPP_MIN_LOG_LEVEL=2 
# train all models for testing
#model with EV-assistance
echo '======================================================'
echo "Training T-trecx Model with EV-assistance....."
echo '======================================================'
sleep 2
python train_mobnet.py 'model_with_ev_assist' 'mobnet_ev'
#model without EV-assistance
echo '======================================================'
echo "Training T-trecx Model without EV-assistance....."
echo '======================================================'
sleep 2
python train_mobnet.py 'model_without_ev_assist' 'mobnet_noev'
