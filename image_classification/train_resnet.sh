#!/bin/bash
# Download the dataset.
# wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
# tar -xvf cifar-10-python.tar.gz

export TF_CPP_MIN_LOG_LEVEL=2 
# train all models for testing
#model with EV-assistance
echo '======================================================'
echo "Training T-trecx Model with EV-assistance....."
echo '======================================================'
sleep 2
python train_resnet.py --isTrecx --isEV --model_architecture=resnet_ev --model_save_name=model_with_ev_assist
#model without EV-assistance
echo '======================================================'
echo "Training T-trecx Model without EV-assistance....."
echo '======================================================'
sleep 2
python train_resnet.py --isTrecx --model_architecture=resnet_noev --model_save_name=model_without_ev_assist

#Resnet with SDN techniques
echo '======================================================'
echo "Training SDN-Resnet....."
echo '======================================================'
sleep 2
python train_resnet.py --model_architecture=resnet_sdn --model_save_name=model_sdn
#Resnet with Branchynet techniques
echo '======================================================'
echo "Training Branchynet-Resnet....."
echo '======================================================'
sleep 2
python train_resnet.py --model_architecture=resnet_branchynet --model_save_name=model_branchynet
