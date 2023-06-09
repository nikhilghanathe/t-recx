
# Image Classification with Resnet-8 on CIFAR-10

A ResNet8 model is trained on the CIFAR10 dataset available at:
https://www.cs.toronto.edu/~kriz/cifar.html

Model: ResNet8

Dataset: CIFAR-10

## Usage

Run the following commands to download dataset and reproduce results from paper.

``` Bash
# Download dataset
./download_dataset.sh (takes a few minutes)

# Reproduce results (15-20 minutes)
python test_resnet.py

```

To train the models,
```Bash
# Train Resnet-8 models (takes several hours)
./train_resnet.sh 

```

## Description
The python format CIFAR10 dataset batches are stored in the __/cifar-10-batches-py__ folder.

The pretrained models can be found in `trained_models/`

The benefit curve presented in the paper are generated and saved to `results/` 

Running `python test_resnet.py` generates benefit curve plots and stores them in `results/`

Running `./train_resnet.sh` will train four versions of the Resnet-8 model 
>* with Early-view (EV) assistance
>* without EV-assistance 
>* with [SDN](https://arxiv.org/abs/1810.07052) techniques
>* with [Branchynet](https://arxiv.org/abs/1709.01686) techniques. 

Each model trains for around 5-6 hours with a GPU. 




