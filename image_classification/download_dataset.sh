#!/bin/bash
# Download the dataset.
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvf cifar-10-python.tar.gz
export TF_CPP_MIN_LOG_LEVEL=2 