#!/bin/bash
# Downoad the dataset.
echo "========================================================"
echo "Downloading and preparing dataset... This may take 20-30min"
echo "========================================================"
sleep 3

wget https://www.silabs.com/public/files/github/machine_learning/benchmarks/datasets/vw_coco2014_96.tar.gz
tar -xvf vw_coco2014_96.tar.gz

#download val2017 dataset (minival) and apply preprocessing
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
python buildPersonDetectionDatabase.py