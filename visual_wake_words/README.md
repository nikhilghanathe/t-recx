# Human Presence Detection on Visual Wake Words Dataset

A MobilenetV1 model is trained on the Visual Wake Words dataset and early-exit is applied to it. The model performs a binary classification task of human face detection. It detects the presence of a person if the person occupies more than 2.5% of the input image

Model: MobilenetV1

Dataset: Visual Wake words

## Usage

Run the following commands to download dataset and reproduce results from the paper. Downloading + preprocessing will require 20-30min. Testing using pretrained models will require 10-15 min.
``` Bash
# Download dataset
./download_and_prepare_dataset.sh
# Reproduce results
python test_mobnet.py
```

To train the models,
```Bash
# Train Mobilenet models (takes several hours)
./train_vww.sh 
```

## Description

The pretrained models can be found in `trained_models/`

The benefit curve presented in the paper are generated and saved to `results/` 

Running `./train_vww.sh`,
* Trains two version of Mobilenet model 
>* with Early-view (EV) assistance 
>* without EV-assistance. 

Each model trains for around 4-5 hours with a GPU. The models are saved in `trained_models/`

Running `python test_mobnet.py` generates benefit curve plots and stores them in `results/`
