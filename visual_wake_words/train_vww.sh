#!/bin/bash
# # Downoad the dataset.
# wget https://www.silabs.com/public/files/github/machine_learning/benchmarks/datasets/vw_coco2014_96.tar.gz
# tar -xvf vw_coco2014_96.tar.gz

# #download val2017 dataset (minival) and apply preprocessing
# wget http://images.cocodataset.org/zips/val2017.zip
# unzip val2017.zip
# wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
# unzip annotations_trainval2017.zip
# python buildPersonDetectionDatabase.py

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
